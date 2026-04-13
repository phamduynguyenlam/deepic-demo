import torch
import numpy as np
from collections import deque
from optimizer.learnable_optimizer import Learnable_Optimizer


def generate_random_int(NP: int, cols: int) -> torch.Tensor:
	r = torch.randint(0, NP, (NP, cols), dtype=torch.long)  # [NP, 3]

	for i in range(NP):
		while r[i, :].eq(i).any():
			r[i, :] = torch.randint(0, NP, (cols,), dtype=torch.long)

	return r


class Surr_RLDE_Optimizer(Learnable_Optimizer):
	def __init__(self, config):
		super().__init__(config)


		self.config = config
		config.F = 0.5
		config.Cr = 0.7
		config.NP = 100
		self.device = config.device
		self.config = config
		self.F = config.F
		self.Cr = config.Cr
		self.pop_size = config.NP
		self.maxFEs = config.maxFEs
		self.dim = config.dim
		self.ub = config.upperbound
		self.lb = -config.upperbound
		# 种群与适应度
		self.population = None
		self.fitness = None
		self.pop_cur_best = None
		self.fit_cur_best = None
		self.pop_history_best = None
		self.fit_history_best = None
		self.fit_init_best = None
		self.improved_gen = 0

		self.fes = None  # record the number of function evaluations used
		self.cost = None
		self.cur_logpoint = None  # record the current logpoint
		self.log_interval = config.log_interval

	def get_state(self, problem):
		state = torch.zeros(9)
		# state 1
		diff = self.population.unsqueeze(0) - self.population.unsqueeze(1)
		distances = torch.sqrt(torch.sum(diff ** 2, dim=2))
		state[0] = torch.sum(distances) / (self.population.shape[0] * (self.population.shape[0] - 1))

		# state 2
		diff = self.population - self.pop_cur_best
		distances = torch.sqrt(torch.sum(diff ** 2, dim=1))
		state[1] = torch.sum(distances) / (self.population.shape[0])

		# state 3
		diff = self.population - self.pop_history_best
		distances = torch.sqrt(torch.sum(diff ** 2, dim=1))
		state[2] = torch.sum(distances) / (self.population.shape[0])

		# state 4
		diff = self.fitness - self.fit_history_best
		distances = torch.sqrt(torch.sum(diff ** 2, dim=-1))
		state[3] = torch.sum(distances) / (self.fitness.shape[0])

		# state 5
		diff = self.fitness - self.fit_cur_best
		# print

		distances = torch.sqrt(torch.sum(diff ** 2, dim=-1))
		state[3] = torch.sum(distances) / (self.fitness.shape[0])

		# state 6 std(y)
		state[5] = torch.std(self.fitness)

		# state 7 (T - t)/T
		state[6] = (self.maxFEs - self.fes) / self.maxFEs

		# state 8
		if self.fit_cur_best < self.fit_history_best:
			self.improved_gen = 0
		else:
			self.improved_gen += 1

		state[7] = self.improved_gen

		# state 9 bool
		if self.fit_cur_best < self.fit_history_best:
			state[8] = 1
		else:
			state[8] = 0
		return state

	def init_population(self, problem):
		self.population = torch.rand(self.pop_size, self.dim) * (problem.ub - problem.lb) + problem.lb
		#(-5,5)
		self.population = self.population.to(self.device)

		if self.config.is_train:
			self.fitness = problem.eval(self.population.clone())

		else:

			if problem.optimum is None:
				self.fitness = problem.eval(self.population.clone().cpu().numpy())
			else:
				self.fitness = problem.eval(self.population.clone().cpu().numpy()) - problem.optimum

		if isinstance(self.fitness, np.ndarray):
			self.fitness = torch.from_numpy(self.fitness).to(self.device)
		if self.fitness.shape == (self.pop_size,):
			self.fitness = self.fitness.unsqueeze(1)

		self.pop_cur_best = self.population[torch.argmin(self.fitness)].clone()
		self.pop_history_best = self.population[torch.argmin(self.fitness)].clone()

		self.fit_init_best = torch.min(self.fitness).clone()
		self.fit_cur_best = torch.min(self.fitness).clone()
		self.fit_history_best = torch.min(self.fitness).clone()

		self.fes = self.pop_size
		self.cost = [self.fit_cur_best.clone().cpu().item()]
		self.cur_logpoint = 1
		state = self.get_state(problem)

		return state

	def update(self, action, problem):

		if action == 0:
			mut_way = 'DE/rand/1'
			self.F = 0.1
		elif action == 1:
			mut_way = 'DE/rand/1'
			self.F = 0.5
		elif action == 2:
			mut_way = 'DE/rand/1'
			self.F = 0.9
		elif action == 3:
			mut_way = 'DE/best/1'
			self.F = 0.1
		elif action == 4:
			mut_way = 'DE/best/1'
			self.F = 0.5
		elif action == 5:
			mut_way = 'DE/best/1'
			self.F = 0.9
		elif action == 6:
			mut_way = 'DE/current-to-rand'
			self.F = 0.1
		elif action == 7:
			mut_way = 'DE/current-to-rand'
			self.F = 0.5
		elif action == 8:
			mut_way = 'DE/current-to-rand'
			self.F = 0.9
		elif action == 9:
			mut_way = 'DE/current-to-pbest'
			self.F = 0.1
		elif action == 10:
			mut_way = 'DE/current-to-pbest'
			self.F = 0.5
		elif action == 11:
			mut_way = 'DE/current-to-pbest'
			self.F = 0.9
		elif action == 12:
			mut_way = 'DE/current-to-best'
			self.F = 0.1
		elif action == 13:
			mut_way = 'DE/current-to-best'
			self.F = 0.5
		elif action == 14:
			mut_way = 'DE/current-to-best'
			self.F = 0.9
		else:
			raise ValueError(f'action error: {action}')

		mut_population = self.mutation(mut_way)
		crossover_population = self.crossover(mut_population)

		if self.config.is_train:
			temp_fit = problem.eval(crossover_population.clone())
		else:

			if problem.optimum is None:
				temp_fit = problem.eval(crossover_population.clone().cpu().numpy())
			else:
				temp_fit = problem.eval(crossover_population.clone().cpu().numpy()) - problem.optimum


		if isinstance(temp_fit, np.ndarray):
			temp_fit = torch.from_numpy(temp_fit).to(self.device)
		if temp_fit.shape == (self.pop_size,):
			temp_fit = temp_fit.unsqueeze(1)

		for i in range(self.pop_size):
			if temp_fit[i].item() < self.fitness[i].item():
				self.fitness[i] = temp_fit[i]
				self.population[i] = crossover_population[i]

		reward = self.fit_history_best > torch.min(self.fit_history_best, torch.min(self.fitness).clone())

		best_index = torch.argmin(self.fitness)

		self.pop_cur_best = self.population[best_index].clone()
		self.fit_cur_best = self.fitness[best_index].clone()
		next_state = self.get_state(problem)

		if self.fit_cur_best < self.fit_history_best:
			self.fit_history_best = self.fit_cur_best.clone()
			self.pop_history_best = self.pop_cur_best.clone()

		is_done = (self.fes >= self.maxFEs)

		self.fes += self.pop_size

		if self.fes >= self.cur_logpoint * self.config.log_interval:
			self.cur_logpoint += 1
			self.cost.append(self.fit_history_best.clone().cpu().item())

		if is_done:
			if len(self.cost) >= self.config.n_logpoint + 1:
				self.cost[-1] = self.fit_history_best.clone().cpu().item()
			else:
				self.cost.append(self.fit_history_best.clone().cpu().item())

		return next_state, reward.item(), is_done

	def mutation(self, mut_way):
		mut_population = torch.zeros_like(self.population, device=self.device)

		if mut_way == 'DE/rand/1':

			r = generate_random_int(self.pop_size, 3)  # Shape: [pop_size, 3]
			a = self.population[r[:, 0]]
			b = self.population[r[:, 1]]
			c = self.population[r[:, 2]]

			v = a + self.F * (b - c)

			v = torch.clamp(v, min=self.lb, max=self.ub)
			mut_population = v

		elif mut_way == 'DE/best/1':
			r = generate_random_int(self.pop_size, 2)
			a = self.population[r[:, 0]]
			b = self.population[r[:, 1]]
			v = self.pop_cur_best + self.F * (a - b)
			v = torch.clamp(v, self.lb, self.ub)
			mut_population = v

		elif mut_way == 'DE/current-to-rand':
			r = generate_random_int(self.pop_size, 3)
			a = self.population[r[:, 0]]
			b = self.population[r[:, 1]]
			c = self.population[r[:, 2]]
			v = self.population + self.F * (a - self.population) + self.F * (b - c)
			v = torch.clamp(v, self.lb, self.ub)
			mut_population = v

		elif mut_way == 'DE/current-to-pbest':
			p = 0.1
			p_num = max(1, int(p * self.pop_size))
			sorted_indices = torch.argsort(self.fitness.clone().flatten())
			pbest_indices = sorted_indices[:p_num]
			r = generate_random_int(self.pop_size, 2)

			a = self.population[r[:, 0]]
			b = self.population[r[:, 1]]

			pbest_index = pbest_indices[torch.randint(0, p_num, (self.pop_size,))]
			pbest = self.population[pbest_index]

			v = self.population + self.F * (pbest - self.population) + self.F * (a - b)
			v = torch.clamp(v, self.lb, self.ub)
			mut_population = v

		elif mut_way == 'DE/current-to-best':
			r = generate_random_int(self.pop_size, 4)
			a = self.population[r[:, 0]]
			b = self.population[r[:, 1]]
			c = self.population[r[:, 2]]
			d = self.population[r[:, 3]]
			v = self.population + self.F * (self.pop_cur_best - self.population) + self.F * (a - b) + self.F * (c - d)
			v = torch.clamp(v, self.lb, self.ub)
			mut_population = v

		else:
			raise ValueError(f'mutation error: {mut_way} is not defined')

		mut_population = torch.tensor(mut_population, device=self.device)
		return mut_population

	def crossover(self, mut_population):
		crossover_population = self.population.clone()
		for i in range(self.pop_size):

			select_dim = torch.randint(0, self.dim, (1,))
			for j in range(self.dim):
				if torch.rand(1) < self.Cr or j == select_dim:
					crossover_population[i][j] = mut_population[i][j]
		return crossover_population


class SAEA(Learnable_Optimizer):
	"""Surrogate-Assisted Evolutionary Algorithm (SAEA).
	
	Archive-based multi-objective optimizer with generation, surrogate prediction, agent-driven real evaluations,
	and non-dominated sorting selection.
	"""

	def __init__(self,
		 config,
		 surrogate_model,
		 agent=None,
		 archive_size: int = 100,
		 offspring_size: int = 100,
		 k_real: int = 10,
		 generation_strategy: str = 'NSGA-III'):
		super().__init__(config)
		self.config = config
		self.surrogate_model = surrogate_model
		self.agent = agent
		self.archive_size = archive_size
		self.offspring_size = offspring_size
		self.k_real = k_real
		self.generation_strategy = generation_strategy
		self.archive_X = None  # np.ndarray [archive_size, dim]
		self.archive_y = None  # np.ndarray [archive_size, n_obj]

	def init_population(self, problem):
		"""Initialize archive with random solutions and their real objective values."""
		if not hasattr(problem, 'dim'):
			self.dim = self.config.dim
		else:
			self.dim = problem.dim

		lb = np.asarray(problem.lb if hasattr(problem, 'lb') else -self.config.upperbound)
		ub = np.asarray(problem.ub if hasattr(problem, 'ub') else self.config.upperbound)
		if lb.shape == ():
			lb = np.full((self.dim,), lb)
		if ub.shape == ():
			ub = np.full((self.dim,), ub)

		self.archive_X = np.random.uniform(lb, ub, size=(self.archive_size, self.dim))
		self.archive_y = problem.eval(self.archive_X)

		self.archive_X, self.archive_y = self._env_rank_and_trim(self.archive_X, self.archive_y)
		return {'X': self.archive_X, 'y': self.archive_y}

	def update(self, action, problem):
		"""One optimization iteration for SAEA."""
		# 1) Generate offspring from archive
		offspring_X = self._generate_offspring(self.archive_X, problem, strategy=self.generation_strategy)

		# 2) Surrogate-approximated objective values
		if hasattr(self.surrogate_model, 'predict'):
			offspring_y_pred = self.surrogate_model.predict(offspring_X)
		else:
			raise AttributeError('surrogate_model must support predict(X)')

		# 3) Select k candidates to real-evaluate
		if self.agent is not None and hasattr(self.agent, 'select'):
			idx = self.agent.select(
				offspring_X,
				offspring_y_pred,
				self.k_real,
				archive_x=self.archive_X,
				archive_y=self.archive_y,
				lower=np.asarray(problem.lb if hasattr(problem, 'lb') else -self.config.upperbound),
				upper=np.asarray(problem.ub if hasattr(problem, 'ub') else self.config.upperbound),
				progress=(self.fes / max(self.maxFEs, 1)) if self.maxFEs else 0.0,
			)
		else:
			# default: choose lowest predicted objective for minimization using first objective
			if offspring_y_pred.ndim == 1:
				idx = np.argsort(offspring_y_pred)[:self.k_real]
			else:
				idx = np.argsort(offspring_y_pred[:, 0])[:self.k_real]

		idx = np.asarray(idx, dtype=int)
		idx = np.unique(idx)
		idx = idx[:min(len(idx), self.k_real)]

		real_X = offspring_X[idx]
		real_y = problem.eval(real_X)

		# 4) Merge into archive and select best by non-dominated sorting
		self.archive_X = np.vstack([self.archive_X, real_X])
		self.archive_y = np.vstack([self.archive_y, real_y])
		self.archive_X, self.archive_y = self._env_rank_and_trim(self.archive_X, self.archive_y)

		return {'X': self.archive_X, 'y': self.archive_y}

	def run(self, problem, iterations: int = 1):
		"""Run the SAEA for a number of iterations and return Pareto archive."""
		if self.archive_X is None or self.archive_y is None:
			self.init_population(problem)

		for _ in range(iterations):
			self.update(None, problem)

		return {'X': self.archive_X, 'y': self.archive_y}

	def _generate_offspring(self, X, problem, strategy='NSGA-III'):
		"""Generate offspring from archive according to strategy."""
		n = self.offspring_size
		if strategy == 'NSGA-III':
			idx = np.random.choice(len(X), size=n, replace=True)
			children = X[idx] + np.random.normal(scale=0.1, size=(n, self.dim))
		elif strategy == 'CDM-PSL':
			idx = np.random.choice(len(X), size=n, replace=True)
			children = X[idx] + np.random.normal(scale=0.2, size=(n, self.dim))
		elif strategy == 'qNEHVI':
			idx = np.random.choice(len(X), size=n, replace=True)
			children = X[idx] + np.random.normal(scale=0.05, size=(n, self.dim))
		else:
			raise ValueError(f'Unknown generation strategy {strategy}')

		# keep in bounds
		lb = np.asarray(problem.lb if hasattr(problem, 'lb') else -self.config.upperbound)
		ub = np.asarray(problem.ub if hasattr(problem, 'ub') else self.config.upperbound)
		if lb.shape == ():
			lb = np.full((self.dim,), lb)
		if ub.shape == ():
			ub = np.full((self.dim,), ub)
		children = np.minimum(np.maximum(children, lb), ub)
		return children

	def _is_dominated(self, y1, y2):
		"""Return True if y1 dominated by y2 in minimization context."""
		return np.all(y2 <= y1) and np.any(y2 < y1)

	def _pareto_front_indices(self, Y):
		"""Compute non-dominated front indices."""
		N = Y.shape[0]
		is_dom = np.zeros(N, dtype=bool)
		for i in range(N):
			for j in range(N):
				if i != j and self._is_dominated(Y[i], Y[j]):
					is_dom[i] = True
					break
		return np.where(~is_dom)[0]

	def _env_rank_and_trim(self, X, Y):
		"""Keep non-dominated solutions and trim to archive size."""
		if Y.ndim == 1:
			Y = Y.reshape(-1, 1)

		front = self._pareto_front_indices(Y)
		X_front = X[front]
		Y_front = Y[front]

		if len(front) > self.archive_size:
			# simple truncation by first-come; could be replaced by crowding distance
			X_front = X_front[:self.archive_size]
			Y_front = Y_front[:self.archive_size]

		return X_front, Y_front
