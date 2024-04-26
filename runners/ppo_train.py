from __future__ import division

import time
import numpy as np
from datasets.data import get_data
from datasets.glove import Glove

import setproctitle

from models.model_io import ModelOptions

from agents.random_agent import RandomNavigationAgent


import random

from .train_util import (
    compute_loss,
    new_episode,
    run_episode,
    transfer_gradient_from_player_to_shared,
    end_episode,
    reset_player,
)

def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]
        
        

def ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, model, clip_param=0.2):
    for _ in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
            dist, value = model(state)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def nonadaptivea3c_train(
    rank,
    args,
    create_shared_model,
    shared_model,
    initialize_agent,
    optimizer,
    res_queue,
    end_flag,
):
    

    glove = Glove(args.glove_file)
    scenes, possible_targets, targets = get_data(args.scene_types, args.train_scenes)

    random.seed(args.seed + rank)
    idx = [j for j in range(len(args.scene_types))]
    random.shuffle(idx)

    setproctitle.setproctitle("Training Agent: {}".format(rank))

    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]

    import torch

    torch.cuda.set_device(gpu_id)

    torch.manual_seed(args.seed + rank)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + rank)

    player = initialize_agent(create_shared_model, args, rank, gpu_id=gpu_id)
    compute_grad = not isinstance(player, RandomNavigationAgent)

    model_options = ModelOptions()

    j = 0

    while not end_flag.value:

        # Get a new episode.
        total_reward = 0
        player.eps_len = 0
        new_episode(
            args, player, scenes[idx[j]], possible_targets, targets[idx[j]], glove=glove
        )
        player_start_time = time.time()

        # Train on the new episode.
        while not player.done:
            # Make sure model is up to date.
            player.sync_with_shared(shared_model)
            # Run episode for num_steps or until player is done.
            total_reward = run_episode(player, args, total_reward, model_options, True)
            # Compute the loss.
            loss = compute_loss(args, player, gpu_id, model_options)
            if compute_grad:
                # Compute gradient.
                player.model.zero_grad()
                loss["total_loss"].backward()
                torch.nn.utils.clip_grad_norm_(player.model.parameters(), 100.0)
                # Transfer gradient to shared model and step optimizer.
                transfer_gradient_from_player_to_shared(player, shared_model, gpu_id)
                optimizer.step()
                # Clear actions and repackage hidden.
            if not player.done:
                reset_player(player)

        for k in loss:
            loss[k] = loss[k].item()

        end_episode(
            player,
            res_queue,
            title=args.scene_types[idx[j]],
            total_time=time.time() - player_start_time,
            total_reward=total_reward,
        )
        reset_player(player)

        j = (j + 1) % len(args.scene_types)

    player.exit()
