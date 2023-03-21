#!/usr/bin/env python3
from estimators import *

# For generating and running test cases
import random, string
import multiprocessing, multiprocessing.pool

# For plotting test results
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Parallel jobs to start, simulation range and number of increments to run
# Default is 50 steps from 1 to 10K unique elements, 100 evaluations per step
NUM_WORKERS = 10
(LOWER_BOUND,UPPER_BOUND) = (1,10000)
STEPS = 50
RUNS = 100

# Yields 'n' random strings of length ten
def generateRandomStrings(n):
	strings = []
	for i in range(0, n):
		strings.append(''.join(random.choices(string.ascii_lowercase, k=10)))
	return strings

# Evaluate a single test case with all three estimator algorithms, return
# a tuple of results from them all
def runTest(args):
	(elements, trial) = args
	strings = generateRandomStrings(elements)
	pc = probabilisticCount(strings)
	pca = probabilisticCountAveraged(strings, avg="median")
	hll = hyperLogLog(strings)
	return [trial,elements,pc,pca,hll]

if __name__ == "__main__":
	results = []
	tasks = []

	# This will take a while, so spin up a process pool to run all our experiments
	pool = multiprocessing.get_context("spawn").Pool(processes=NUM_WORKERS)
	for elements in np.linspace(LOWER_BOUND,UPPER_BOUND,STEPS,dtype=int):
		for trial in range(1, RUNS):
			tasks.append((elements,trial))
	results = list(pool.imap_unordered(runTest, tasks))
	pool.close()
	pool.join()
	pool.terminate()

	# Load results into a Seaborn-friendly shape
	df = pd.DataFrame(results, columns=["Trial", "Elements", "Probabilistic", "Probabilistic-Med", "HyperLogLog"])
	df = pd.melt(df, id_vars=["Trial", "Elements"], var_name="Estimator", value_name="Estimated Unique Elements")

	# Plot and save
	sns.lineplot(data=df, x="Elements", y="Estimated Unique Elements", hue="Estimator")
	sns.despine()
	# Cut off huge outliers from Probabilistic count
	plt.ylim(LOWER_BOUND, UPPER_BOUND)
	plt.xlabel("True Unique Elements")
	plt.title("Accuracy of Unique Count Estimator Functions")
	plt.savefig("trials.svg", bbox_inches="tight", dpi=200)
