package kricket.neural;

import java.util.List;

import kricket.neural.util.Datum;
import kricket.neural.util.NNOptions;

public abstract class NNBase {
	
	protected NNOptions options;
	
	public NNBase(NNOptions opts) {
		options = opts;
	}

	public NNOptions getOptions() {
		return options;
	}
	
	/**
	 * Perform Stochastic Gradient Descent using the given data.
	 * @param trainingSet The training data.
	 * @param batchSize The size of each mini-batch to use.
	 * @param epochs The number of training epochs.
	 * @param eta The training rate.
	 * @param lambda The regularization parameter (for L2 regularization - set to 0 to ignore).
	 */
	public void SGD(List<? extends Datum> trainingSet, int batchSize, int epochs, double eta, double lambda) {
		if(options.summarizeSGD)
			options.log.info("Performing SGD with:\n\tNum data: " + trainingSet.size()
				+ "\n\tBatch size: " + batchSize
				+ "\n\tEpochs: " + epochs
				+ "\n\tTraining rate: " + eta
				+ "\n\tRegularization rate: " + lambda
				);

		double regTerm = (lambda == 0 ? 0 : 1 - (eta*lambda / trainingSet.size()));
		
		for(int epoch = 0; epoch < epochs; epoch++) {
			if(options.logEpochs)
				options.log.info("Running epoch " + epoch);
			
			long startTime = System.currentTimeMillis();
			
			for(int start=0; start<=trainingSet.size()-batchSize; start+=batchSize) {
				runBatch(trainingSet.subList(start, start+batchSize), regTerm, eta);
			}
			
			if(options.logEpochs)
				options.log.info(String.format("Epoch completed in %.3fs", (System.currentTimeMillis() - startTime)*0.001));
			
			// How did we do?
			if(options.calcErrorsAfterEpochs)
				calc_error(trainingSet);
		}
	}

	/**
	 * Run a batch as part of SGD.
	 * @param batch
	 * @param regTerm L2-regularization term (0 = ignore)
	 * @param eta Training rate.
	 */
	protected abstract void runBatch(List<? extends Datum> batch, double regTerm, double eta);
	
	/**
	 * Get the % error of the network with the given data.
	 * @param data
	 * @return
	 */
	public abstract double calc_error(List<? extends Datum> data);
	
	protected boolean isCorrect(double[] result, double[] answer) {
		int guess = 0;
		double max = result[0];
		for(int i=1; i<result.length; i++) {
			if(result[i] > max) {
				max = result[i];
				guess = i;
			}
		}
		return answer[guess] > 0.99;
	}
}
