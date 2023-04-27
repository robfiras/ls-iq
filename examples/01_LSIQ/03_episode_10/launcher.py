from experiment_launcher import Launcher

from experiment_launcher.utils import bool_local_cluster

if __name__ == '__main__':
    LOCAL = bool_local_cluster()
    TEST = False
    USE_CUDA = False

    JOBLIB_PARALLEL_JOBS = 1  # or os.cpu_count() to use all cores
    N_SEEDS = 5

    launcher = Launcher(exp_name='lsiq_10',
                        python_file='lsiq_experiments',
                        n_exps=N_SEEDS,
                        joblib_n_jobs=JOBLIB_PARALLEL_JOBS,
                        n_cores=JOBLIB_PARALLEL_JOBS * 1,
                        memory_per_core=JOBLIB_PARALLEL_JOBS * 6000,
                        days=2,
                        hours=0,
                        minutes=0,
                        seconds=0,
                        use_timestamp=True,
                        )

    default_params = dict(n_epochs=150,
                          n_steps_per_epoch=10000,
                          n_eval_episodes=10,
                          n_steps_per_fit=1,
                          n_epochs_save=-1,
                          logging_iter=10000,
                          gamma=0.99,
                          use_cuda=USE_CUDA,
                          tau=0.005,
                          use_target=True,
                          loss_mode_exp="fix",
                          regularizer_mode="plcy",
                          learnable_alpha=False)

    log_std = [(-5, 2)]
    envs = ["Ant-v3",
            "HalfCheetah-v3",
            "Hopper-v3",
            "Humanoid-v3",
            "Walker2d-v3"]
    path_to_datasets = "../../00_Datasets/10_episodes/"
    expert_data_filenames = ["expert_dataset_Ant-v3_6421.34_10_SAC.npz",
                             "expert_dataset_HalfCheetah-v3_12360.31_10_SAC.npz",
                             "expert_dataset_Hopper-v3_3549.94_10_SAC.npz",
                             "expert_dataset_Humanoid-v3_6346.43_10_SAC.npz",
                             "expert_dataset_Walker2d-v3_5852.24_10_SAC.npz"]

    expert_data_paths = [path_to_datasets + name for name in expert_data_filenames]

    # Ant
    launcher.add_experiment(env_id__=envs[0], expert_data_path=expert_data_paths[0],
                            plcy_loss_mode__="value", init_alpha__=1e-3, Q_exp_loss__="MSE", reg_mult__=0.5, **default_params)


    # HalfCheetah
    launcher.add_experiment(env_id__=envs[1], expert_data_path=expert_data_paths[1],
                            plcy_loss_mode__="value", init_alpha__=1e-3, Q_exp_loss__="MSE", reg_mult__=0.5, **default_params)
    launcher.add_experiment(env_id__=envs[1], expert_data_path=expert_data_paths[1],
                            plcy_loss_mode__="value", init_alpha__=1e-3, Q_exp_loss__="MSE", reg_mult__=10.0, **default_params)

    # Hopper
    launcher.add_experiment(env_id__=envs[2], expert_data_path=expert_data_paths[2],
                            plcy_loss_mode__="value", init_alpha__=1e-3, Q_exp_loss__="MSE", reg_mult__=0.5, **default_params)

    # Humanoid
    launcher.add_experiment(env_id__=envs[3], expert_data_path=expert_data_paths[3],
                            plcy_loss_mode__="value", init_alpha__=0.1, Q_exp_loss__="MSE", reg_mult__=0.5, **default_params)

    # Walker2d
    launcher.add_experiment(env_id__=envs[4], expert_data_path=expert_data_paths[4],
                            plcy_loss_mode__="value", init_alpha__=1e-3, Q_exp_loss__="MSE", reg_mult__=0.5, **default_params)

    launcher.run(LOCAL, TEST)
