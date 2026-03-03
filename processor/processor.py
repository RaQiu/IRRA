
import logging
import time
import torch
from torch.utils.tensorboard import SummaryWriter

from utils.comm import get_rank, synchronize
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from modality_grad_modulator import ModalityGradModulator, ModulationConfig


def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer):
    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("IRRA.train")
    logger.info('start training')

    meters = {
        "loss": AverageMeter(),
        "sdm_loss": AverageMeter(),
        "itc_loss": AverageMeter(),
        "id_loss": AverageMeter(),
        "mlm_loss": AverageMeter(),
        "img_acc": AverageMeter(),
        "txt_acc": AverageMeter(),
        "mlm_acc": AverageMeter(),
        "rho_t": AverageMeter()
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_top1 = 0.0

    # ---- Modulation plugin setup ----
    config = ModulationConfig(
        tau=int(args.tau),
        gammb=args.gammb,
        gaeta=args.gaeta,
        alpha=args.alpha,
        beta=args.beta,
        enabled=args.modulation,
        fig1c_enabled=True,
        fig1c_noise_std=0.1,
        output_dir=args.output_dir,
    )

    def shared_filter(name, mod):
        """Cross-modal shared modules (hooks + neuron-level pen)."""
        return (not name.startswith('base_model')
                and not name.endswith('classifier')
                and not name.startswith('ln_pre_'))

    def img_enc_filter(name, mod):
        """Image encoder leaf modules (encoder-level suppression)."""
        return name.startswith(('base_model.visual', 'ln_pre_i'))

    def txt_enc_filter(name, mod):
        """Text encoder leaf modules (encoder-level suppression)."""
        return name.startswith(('base_model.transformer', 'ln_pre_t'))

    modulator = ModalityGradModulator(config)
    modulator.attach(model, shared_filter, img_enc_filter, txt_enc_filter)

    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()

        modulator.on_epoch_start(model)
        model.train()
        step_size = len(train_loader)

        for n_iter, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            # ---- Three forward passes with activation capture ----
            modulator.pre_forward(model)

            ret = model(batch)
            modulator.capture('normal')

            with torch.no_grad():
                e_txt_batch = batch.copy()
                e_txt_ret = model(e_txt_batch, erase_modality="e_txt")
                modulator.capture('e_txt')

                e_img_batch = batch.copy()
                e_img_ret = model(e_img_batch, erase_modality="e_img")
                modulator.capture('e_img')

            total_loss = sum([v for k, v in ret.items() if "loss" in k])
            total_mlm_loss = ret['mlm_loss']
            e_img_mlm_loss = e_img_ret['mlm_loss']
            e_txt_mlm_loss = e_txt_ret['mlm_loss']

            batch_size = batch['images'].shape[0]
            meters['loss'].update(total_loss.item(), batch_size)
            meters['sdm_loss'].update(ret.get('sdm_loss', 0), batch_size)
            meters['itc_loss'].update(ret.get('itc_loss', 0), batch_size)
            meters['id_loss'].update(ret.get('id_loss', 0), batch_size)
            meters['mlm_loss'].update(ret.get('mlm_loss', 0), batch_size)

            meters['img_acc'].update(ret.get('img_acc', 0), batch_size)
            meters['txt_acc'].update(ret.get('txt_acc', 0), batch_size)
            meters['mlm_acc'].update(ret.get('mlm_acc', 0), 1)

            optimizer.zero_grad()
            total_loss.backward()

            # ===== rho_t: Q/K gradient Frobenius norm ratio =====
            if args.distributed:
                cross_attn = model.module.cross_attn
            else:
                cross_attn = model.cross_attn

            embed_dim = model.embed_dim if not args.distributed else model.module.embed_dim
            in_proj_weight = cross_attn.in_proj_weight

            if in_proj_weight.grad is not None:
                grad_WQ = in_proj_weight.grad[:embed_dim, :]
                grad_WK = in_proj_weight.grad[embed_dim:2*embed_dim, :]
                norm_WQ = torch.norm(grad_WQ, p='fro')
                norm_WK = torch.norm(grad_WK, p='fro')
                rho_t = norm_WQ / (norm_WK + 1e-8)
            else:
                rho_t = torch.tensor(0.0)

            meters['rho_t'].update(rho_t.item(), batch_size)

            # ---- Gradient modulation via plugin ----
            modulator.post_backward(
                model, step=n_iter, data_len=step_size,
                batch_size=batch_size,
                all_loss=total_mlm_loss,
                e_txt_loss=e_txt_mlm_loss,
                e_img_loss=e_img_mlm_loss,
            )

            del e_txt_batch, e_img_batch
            torch.cuda.empty_cache()

            optimizer.step()
            synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                info_str += f", rho_t: {meters['rho_t'].avg:.4f}"
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)

        # ---- Epoch end: fig1c save + modal count logging + reset ----
        stats = modulator.on_epoch_end(model, epoch)

        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        tb_writer.add_scalar('rho_t', meters['rho_t'].avg, epoch)
        for k, v in meters.items():
            if v.avg > 0:
                tb_writer.add_scalar(k, v.avg, epoch)

        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                if args.distributed:
                    top1 = evaluator.eval(model.module.eval())
                else:
                    top1 = evaluator.eval(model.eval())

                torch.cuda.empty_cache()
                if best_top1 < top1:
                    best_top1 = top1
                    arguments["epoch"] = epoch
                    checkpointer.save("best", **arguments)

    if get_rank() == 0:
        logger.info(f"best R1: {best_top1} at epoch {arguments['epoch']}")


def do_inference(model, test_img_loader, test_txt_loader):
    logger = logging.getLogger("IRRA.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader)
    top1 = evaluator.eval(model.eval())
