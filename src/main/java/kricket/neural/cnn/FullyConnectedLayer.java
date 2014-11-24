package kricket.neural.cnn;

import kricket.neural.util.Dimension;
import kricket.neural.util.IncompatibleLayerException;
import kricket.neural.util.Matrix;

public class FullyConnectedLayer extends Layer {
	
	private final Matrix weights, biases;
	private final Matrix[] lastZ, lastActivation;
	/**
	 * Accumulated gradients, calculated during backpropagation.
	 */
	private Matrix nabla_Cw, nabla_Cb;
	
	public FullyConnectedLayer(int inputLength, int outputLength) {
		weights = Matrix.random(outputLength, inputLength);
		biases = Matrix.random(outputLength, 1);
		lastZ = new Matrix[1];
		lastActivation = new Matrix[1];
	}
	
	private Matrix flatten(Matrix[] m) {
		return new Matrix(m[0].data);
	}

	@Override
	public Matrix[] feedForward(Matrix[] featureMaps) {
		if(featureMaps.length != 1)
			throw new IllegalArgumentException(featureMaps.length+" feature maps were given.");
		
		lastZ[0] = weights.times(flatten(featureMaps)).plusEquals(biases);
		lastActivation[0] = sigma(lastZ[0].copy());
		
		return lastActivation;
	}
	
	@Override
	public void calcGradients(Matrix[] prevActivations, Matrix[] deltas) {
		if(deltas.length != 1 || prevActivations.length != 1)
			throw new IllegalArgumentException("WTF? We have "+deltas.length+" deltas and "+prevActivations.length+" previous activations!");
		
		nabla_Cb.plusEquals(deltas[0]);
		nabla_Cw.plusEquals(deltas[0].timesTranspose(flatten(prevActivations)));
	}

	@Override
	public Matrix[] backprop(Matrix[] prevZ, Matrix[] deltas) {
		Matrix backDelta = weights
				.transposeTimes(deltas[0])
				.dotTimesEquals(dSigma(prevZ[0]));
		
		return new Matrix[] {backDelta};
	}

	@Override
	public void resetGradients() {
		nabla_Cb = new Matrix(biases.rows, biases.cols);
		nabla_Cw = new Matrix(weights.rows, weights.cols);
	}

	@Override
	public Matrix[] lastActivation() {
		return lastActivation;
	}

	@Override
	public Matrix[] lastZ() {
		return lastZ;
	}
	
	@Override
	public void applyGradients(double regTerm, double eta, int batchSize) {
		if(regTerm != 0)
			weights.timesEquals(regTerm);
		weights.plusEquals(nabla_Cw.timesEquals(-eta / batchSize));
		biases.plusEquals(nabla_Cb.timesEquals(-eta / batchSize));
	}

	@Override
	public Dimension getOutputDimension(Dimension inputDimension) throws IncompatibleLayerException {
		if(inputDimension.columns != 1 || inputDimension.depth != 1 || inputDimension.rows != weights.cols)
			throw new IncompatibleLayerException(inputDimension, this);
		return new Dimension(biases.rows, 1, 1);
	}
	
	@Override
	public String toString() {
		return getClass().getSimpleName() + " (input " + weights.cols + " => output " + weights.rows + ")";
	}
}
