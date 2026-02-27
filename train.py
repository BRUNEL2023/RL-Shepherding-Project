from envs.shepherd_env import ShepherdEnv
from agents.rl_agent import train_rl_agent_ppo_mlp, train_rl_agent_td3_mlp
from agents.CNN_QN import train_image_dqn, ImageDQNAgent, N_ACTIONS
import torch
import argparse

# Fonction helper pour gérer les booléens en ligne de commande
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description="Train RL agents for ShepherdEnv.")
parser.add_argument("-a", "--algorithm", type=str, choices=["td3", "dqn", "ppo", "all"], default="ppo", help="Algorithm choice.")
parser.add_argument("-s", "--num_sheep", type=int, default=1, help="Number of sheep.")
parser.add_argument("-m", "--max_steps", type=int, default=500, help="Max steps per episode.")
parser.add_argument("-r", "--obstacle_radius", type=float, default=0.0, help="Radius of obstacles (Level 3).")
parser.add_argument("-g", "--goal_radius", type=float, default=0.7, help="Radius of goal.")
parser.add_argument("-t", "--timesteps", type=int, default=2000000, help="Total timesteps.")

# NOUVEAU : Argument pour le Level 2
parser.add_argument("-w", "--wandering", type=float, default=0.0, help="Wandering strength (Level 2 noise).")

# NOUVEAU : Argument checkpoint et curriculum
parser.add_argument("-c", "--checkpoint_dir", type=str, default=None, help="Path to checkpoint for fine-tuning.")
parser.add_argument("-cl", "--criculam_learning", type=str2bool, default=True, help="Curriculum learning (reset optimizer).")

args = parser.parse_args()

# Initialisation ENVIRONNEMENT avec wandering
env = ShepherdEnv(
    n_sheep=args.num_sheep,
    max_steps=args.max_steps,
    obstacle_radius=args.obstacle_radius,
    goal_radius=args.goal_radius,
    wandering_strength=args.wandering # <--- Ajouté
)

eval_env = ShepherdEnv(
    n_sheep=args.num_sheep,
    max_steps=args.max_steps,
    obstacle_radius=args.obstacle_radius,
    goal_radius=args.goal_radius,
    wandering_strength=args.wandering # <--- Ajouté
)

if args.algorithm in ["dqn", "all"]:
    try:
        print(f"Training DQN (#sheep: {env.n_sheep})...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        agent = ImageDQNAgent(n_actions=N_ACTIONS, lr=1e-4, gamma=0.99, device=device)
        # Note: DQN ici n'utilise pas checkpoint_dir dans ce snippet, à implémenter si besoin
        train_image_dqn(env=env, eval_env=eval_env, agent=agent, episodes=1000, batch_size=32, target_update=1000, eval_every=5, eval_episodes=5)
    except Exception as e:
        print(f"DQN training failed: {e}")

if args.algorithm in ["ppo", "all"]:
    try:
        print(f"Training PPO (#sheep: {env.n_sheep}, wandering: {args.wandering}, obstacle: {args.obstacle_radius})...")
        model = train_rl_agent_ppo_mlp(
            env, eval_env, 
            timesteps=args.timesteps,
            checkpoint_dir=args.checkpoint_dir,
            criculam_learning=args.criculam_learning
        )
        # Nom du fichier dynamique
        fname = f"models/ppo_s{env.n_sheep}_w{int(args.wandering*100)}_r{int(args.obstacle_radius*10)}"
        model.save(fname)
        print(f"Model saved to {fname}")
    except Exception as e:
        print(f"PPO training failed: {e}")

if args.algorithm in ["td3", "all"]:
    try:
        print(f"Training TD3 (#sheep: {env.n_sheep})...")
        model = train_rl_agent_td3_mlp(
            env, eval_env, 
            timesteps=args.timesteps,
            checkpoint_dir=args.checkpoint_dir,
            criculam_learning=args.criculam_learning
        )
        fname = f"models/td3_s{env.n_sheep}_w{int(args.wandering*100)}_r{int(args.obstacle_radius*10)}"
        model.save(fname)
    except Exception as e:
        print(f"TD3 training failed: {e}")