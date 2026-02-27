import argparse
import numpy as np
import os
import torch
from stable_baselines3 import PPO, TD3
from envs.shepherd_env import ShepherdEnv
from agents.rule_based_agent import RuleBasedShepherd, TipsyShepherd, LazyShepherd
from agents.CNN_QN import ImageDQNAgent, N_ACTIONS, render_env_to_rgb, ANGLES, transform

def load_agent(agentType: str, model_name: str, env: ShepherdEnv):
    # (Fonction load_agent inchangée, juste s'assurer qu'elle gère bien les chemins)
    if agentType == "ruleBase": return RuleBasedShepherd()
    if agentType == "tipsy": return TipsyShepherd()
    if agentType == "lazy": return LazyShepherd()
    
    if agentType == "DQN" and os.path.exists(model_name):
        print("Using DQN Agent.")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        agent = ImageDQNAgent(n_actions=N_ACTIONS, lr=1e-4, gamma=0.99, device=device)
        agent.q_net.load_state_dict(torch.load(model_name, map_location=device))
        agent.q_net.eval()
        return agent

    if os.path.exists(model_name):
        print(f"Loading {agentType} from {model_name}...")
        return globals()[agentType].load(model_name, env=env, device="cpu")
    else:
        # Si pas de fichier, on retourne None ou erreur
        if agentType in ["PPO", "TD3"]:
             raise ValueError(f"Model file not found: {model_name}")
        raise ValueError(f"Unsupported agent type: {agentType}")

def run_episode(env: ShepherdEnv, agent, model_type: str, display_flag=False) -> float:
    obs = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        if model_type in ["ruleBase","lazy","tipsy"]:
            actions = agent.act(obs)
        elif model_type == "DQN":
            state = transform(render_env_to_rgb(env))
            with torch.no_grad():
                action_idx = agent.select_action(state)
                actions = [ANGLES[action_idx]]
        else: # PPO, TD3
            actions, _ = agent.predict(obs, deterministic=True)

        obs, reward, done, _ = env.step(actions)
        total_reward += reward
        if display_flag:
            env.render()

    return total_reward

def main():
    parser = argparse.ArgumentParser(description="Shepherd Environment Test Runner")
    parser.add_argument("-a", "--agent_dir", type=str, default="models/", help="Path to model.")
    parser.add_argument("-t", "--agentType", type=str, choices=["ruleBase", "lazy", "tipsy", "PPO", "DQN", "TD3"], default="ruleBase", help="Agent type.")
    parser.add_argument("-n", "--num_episodes", type=int, default=1, help="Number of episodes.")
    parser.add_argument("-s", "--num_sheep", type=int, default=1, help="Number of sheep.")
    parser.add_argument("-m", "--max_steps", type=int, default=500, help="Max steps.")
    parser.add_argument("-r", "--obstacle_radius", type=float, default=0.0, help="Obstacle radius.")
    parser.add_argument("-g", "--goal_radius", type=float, default=0.7, help="Goal radius.")
    
    # NOUVEAU : Argument Wandering pour le test
    parser.add_argument("-w", "--wandering", type=float, default=0.0, help="Wandering strength.")

    args = parser.parse_args()
    rewards = []
    successes = 0

    print(f"Running {args.num_episodes} episodes with {args.agentType}...")

    for eps in range(1, args.num_episodes + 1):
        # Initialisation avec wandering
        env = ShepherdEnv(
            n_sheep=args.num_sheep,
            max_steps=args.max_steps,
            obstacle_radius=args.obstacle_radius,
            goal_radius=args.goal_radius,
            wandering_strength=args.wandering # <--- Ajouté
        )
        
        # Charger l'agent (pour RL, on charge à chaque fois pour être propre, ou on pourrait le sortir de la boucle)
        if args.agentType in ["PPO", "TD3", "DQN"]:
            agent = load_agent(args.agentType, args.agent_dir, env)
        else:
            agent = load_agent(args.agentType, "", env)

        final_reward = run_episode(env, agent, args.agentType, display_flag=(args.num_episodes==1))
        rewards.append(final_reward)
        
        # Calcul succès simple (si récompense élevée)
        if final_reward > 1000: # Seuil arbitraire pour succès, à ajuster selon votre reward
            successes += 1
            
        env.close()

    if args.num_episodes > 1:
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        success_rate = (successes / args.num_episodes) * 100.0
        ci = 1.96 * (std_reward / np.sqrt(len(rewards)))

        
        print(f"AGENT           | SUCCESS RATE | AVG REWARD   | AVG STEPS (approx)") # Note: steps pas trackés ici directement, à ajouter si besoin
        print("-----------------------------------------------------------------")
        # Petit fix pour l'affichage, ici je n'ai pas la moyenne des steps, je mets N/A
        print(f"{args.agentType:<15} | {success_rate:>10.1f}% | {mean_reward:>10.2f} | N/A")
        print("=================================================================")
        print(f"Confidence Interval (95% CI): ± {ci:.2f}")

if __name__ == "__main__":
    main()