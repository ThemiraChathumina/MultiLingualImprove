def get_train_ds_config(train_batch_size=1,
                        train_micro_batch_size_per_gpu=1,
                        lr=2e-5,
                        gradient_accumulation_steps=1,
                        offload=True,
                        stage=2,
                        enable_hybrid_engine=False,
                        inference_tp_size=1,
                        release_inference_cache=False,
                        pin_parameters=True,
                        tp_gather_partition_size=8,
                        max_out_tokens=512,
                        warm_step=0,
                        train_step=0):

    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "offload_param": {
            "device": device
        },
        "offload_optimizer": {
            "device": device
        },
        "stage3_param_persistence_threshold": 1e4,
        "stage3_max_live_parameters": 3e7,
        "stage3_prefetch_bucket_size": 3e7,
        "memory_efficient_linear": False
    }
    return {
        "train_batch_size": train_batch_size,
        "train_micro_batch_size_per_gpu": train_micro_batch_size_per_gpu,
        "steps_per_print": 2000,
        "zero_optimization": zero_opt_dict,
        "bf16": {
            "enabled": True,
            "loss_scale_window": 50,
            "min_loss_scale": 1e-10,
        },
        "gradient_clipping": 1.0,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "hybrid_engine": {
            "enabled": enable_hybrid_engine,
            "max_out_tokens": max_out_tokens,
            "inference_tp_size": inference_tp_size,
            "release_inference_cache": release_inference_cache,
            "pin_parameters": pin_parameters,
            "tp_gather_partition_size": tp_gather_partition_size,
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": lr,
                "betas": [
                    0.8,
                    0.999
                ],
                "eps": 1e-8
            }
        },
        "scheduler": {
            "type": "WarmupCosineLR",
            "params": {
            "total_num_steps": train_step,
            "warmup_num_steps": warm_step
            }
        },
    }
