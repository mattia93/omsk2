from tensorflow.keras.utils import Sequence
import numpy as np
import random

class PlanGenerator_old(Sequence):
    def __getitem__(self, index):
        batches = self.plans[index * self.batch_size:(index + 1) * self.batch_size]
        X = np.zeros((int(self.batch_size), int(self.max_dim), len(self.dizionario)))
        Y = np.zeros((int(self.batch_size), len(self.dizionario_goal)))
        for i, plan in enumerate(batches):
            actions = get_actions(plan.actions, self.perc, self.dizionario)
            fill_action_sequence(X, self.max_dim, actions, i)
            Y[i] = get_goal(plan.goals, self.dizionario_goal)
        return X, Y

    def __len__(self):
        return len(self.plans) // self.batch_size

    def __init__(self, plans, dizionario, dizionario_goal, batch_size, max_dim, perc, shuffle=True):
        self.plans = plans
        self.dizionario_goal = dizionario_goal
        self.dizionario = dizionario
        self.batch_size = batch_size
        self.max_dim = max_dim
        self.perc = perc
        self.shuffle = shuffle

    def on_epoch_end(self):
        '''Updates indexes after each epoch'''
        if self.shuffle == True:
            np.random.shuffle(self.plans)


class PlanGenerator(Sequence):
    def __getitem__(self, index):
        batches = self.plans[index * self.batch_size:(index + 1) * self.batch_size]
        X = np.zeros((int(self.batch_size), int(self.max_dim)))
        Y = np.zeros((int(self.batch_size), len(self.dizionario_goal)))
        for i, plan in enumerate(batches):
            actions = get_actions(plan.actions, self.perc, self.dizionario)
            fill_action_sequence(X, self.max_dim, actions, i)
            Y[i] = get_goal(plan.goals, self.dizionario_goal)
        return X, Y

    def __len__(self):
        return len(self.plans) // self.batch_size

    def __init__(self, plans, dizionario, dizionario_goal, batch_size, max_dim, perc, shuffle=True):
        self.plans = plans
        self.dizionario_goal = dizionario_goal
        self.dizionario = dizionario
        self.batch_size = batch_size
        self.max_dim = max_dim
        self.perc = perc
        self.shuffle = shuffle

    def on_epoch_end(self):
        '''Updates indexes after each epoch'''
        if self.shuffle == True:
            np.random.shuffle(self.plans)


class PlanGeneratorMultiPerc(Sequence):
    def __getitem__(self, index):
        batches = self.plans[index * self.batch_size:(index + 1) * self.batch_size]
        X = np.zeros((int(self.batch_size), int(self.max_dim)))
        Y = np.zeros((int(self.batch_size), len(self.dizionario_goal)))
        for i, plan in enumerate(batches):
            seed = plan.plan_name.rsplit('-p',1)[1]
            seed = seed.split('_', 1)[0]
            np.random.seed(int(seed))
            p = np.random.uniform(self.min_perc, self.perc)
            actions = get_actions(plan.actions, p, self.dizionario)
            fill_action_sequence(X, self.max_dim, actions, i)
            Y[i] = get_goal(plan.goals, self.dizionario_goal)
        return X, Y

    def __len__(self):
        return len(self.plans) // self.batch_size

    def __init__(self, plans, dizionario, dizionario_goal, batch_size, max_dim, min_perc, max_perc, shuffle=True):
        self.plans = plans
        self.dizionario_goal = dizionario_goal
        self.dizionario = dizionario
        self.batch_size = batch_size
        self.max_dim = max_dim
        self.min_perc = min_perc
        self.perc = max_perc
        self.shuffle = shuffle

    def on_epoch_end(self):
        '''Updates indexes after each epoch'''
        if self.shuffle == True:
            np.random.shuffle(self.plans)

def get_actions(actions, perc, dizionario):
    size = int(np.ceil(len(actions) * perc))
    indexes = np.ones(size, dtype=int) * -1
    i = 0
    while i < size:
        ind = random.randint(0, len(actions) - 1)
        if ind not in indexes:
            indexes[i] = ind
            i += 1
    indexes = np.sort(indexes)
    return [dizionario[a.name] for a in np.take(actions, indexes)]


def fill_action_sequence(X, max_dim, actions, i):
    for j in range(max_dim):
        if j < len(actions):
            X[i][j] = actions[j]
        else:
            if type(actions[0]) == int:
                X[i][j] = 0
            else:
                X[i][j] = np.zeros(shape=(len(actions[0]),))

def get_goal(g, dizionario_goal):
    goal = np.zeros(len(dizionario_goal))
    for subgoal in g:
        goal = goal + dizionario_goal[subgoal]
    return goal
    
np.random.seed(43)
random.seed(43)

