from stable_baselines3 import PPO, TD3
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
from stable_baselines3.common.callbacks import EvalCallback

def train_rl_agent_ppo_mlp(env, eval_env, timesteps=2000000, checkpoint_dir=None, criculam_learning=False):
    log_dir = f"./logs/ppo/"

    if checkpoint_dir is not None:
        print(f"Loading PPO checkpoint from {checkpoint_dir}...")
        model = PPO.load(
            checkpoint_dir,
            env=env,
            # Paramètres importants à maintenir lors du reload
            n_steps=2048,
            ent_coef=0.003, 
            learning_rate=3e-4,
            gamma=0.99,
            verbose=1,
            tensorboard_log=log_dir
        )
        if criculam_learning:
            print("Curriculum Learning: Reinitializing optimizer...")
            model._setup_model()
            # On peut réduire le LR pour le fine-tuning
            model.learning_rate = 1e-4 
    else:
        print("Training PPO from scratch.")
        model = PPO(
            "MlpPolicy",
            env=env,
            n_steps=2048,
            ent_coef=0.003, # Important pour l'exploration
            learning_rate=3e-4,
            gamma=0.99,
            verbose=1,
            tensorboard_log=log_dir
        )
    
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=log_dir,
        log_path=log_dir, 
        eval_freq=20000,
        n_eval_episodes=10, # Réduit pour accélérer
        deterministic=True, 
        render=False
    )
    
    model.learn(total_timesteps=timesteps, callback=eval_callback)
    return model

def train_rl_agent_td3_mlp(env, eval_env, timesteps=2000000, checkpoint_dir=None, criculam_learning=False):
    log_dir = f"./logs/td3/"
    n_actions = env.action_space.shape[-1]
    # Bruit pour l'exploration TD3
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # NOUVEAU : Gestion du checkpoint pour TD3
    if checkpoint_dir is not None:
        print(f"Loading TD3 checkpoint from {checkpoint_dir}...")
        model = TD3.load(
            checkpoint_dir,
            env=env,
            action_noise=action_noise,
            tensorboard_log=log_dir
        )
        if criculam_learning:
            print("Curriculum Learning: Reinitializing optimizer...")
            model._setup_model()
            model.learning_rate = 1e-4
    else:
        print("Training TD3 from scratch.")
        model = TD3(
            "MlpPolicy", 
            env=env, 
            action_noise=action_noise,
            verbose=1,
            tensorboard_log=log_dir
        )

    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=log_dir,
        log_path=log_dir, 
        eval_freq=20000,
        n_eval_episodes=10,
        deterministic=True, 
        render=False
    )
    
    model.learn(total_timesteps=timesteps, callback=eval_callback)
    return model