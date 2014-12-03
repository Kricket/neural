package kricket.neural.cnn;

import kricket.neural.util.Dimension;
import kricket.neural.util.IncompatibleLayerException;
import kricket.neural.util.Matrix;

/**
 * A fully-connected layer contains a number of neurons. Each neuron's output is a
 * linear function of all the inputs to this layer.
 */
public class FullyConnectedLayer implements Layer {

	/**
	 * The parameters of this Layer.
	 */
	Matrix weights, biases;
	/**
	 * The last input received.
	 */
	Matrix lastX;
	/**
	 * The running total of the calculated gradients of the weights and biases.
	 */
	Matrix dW, dB;
	
	private final int NEURONS;
	
	/**
	 * @param numNeurons The number of neurons in this layer.
	 */
	public FullyConnectedLayer(int numNeurons) {
		NEURONS = numNeurons;
	}
	
	@Override
	public Matrix[] feedForward(Matrix[] x) {
		if(x.length != 1)
			throw new IllegalArgumentException("You tried to send me " + x.length + " feature maps!");
		if(x[0].data.length != weights.cols)
			throw new IllegalArgumentException("Should have " + weights.cols + " inputs, but actually got " + x[0].data.length);
		
		lastX = x[0];
		return new Matrix[] {weights.times(lastX).plusEquals(biases)};
	}

	@Override
	public Matrix[] backprop(Matrix[] deltas) {
		if(deltas[0].data.length != biases.data.length)
			throw new IllegalArgumentException("Should have " + biases.rows + " deltas, but actually got " + deltas[0].data.length);
		
		/*
		 * backprop is actually two operations:
		 * - calculate the derivatives wrt the weights and biases, and add them to dW and dB
		 * - calculate the derivatives wrt the inputs, and return them
		 */
		dB.plusEquals(deltas[0]);
		dW.plusEquals(deltas[0].timesTranspose(lastX));
		
		return new Matrix[] {weights.transposeTimes(deltas[0])};
	}

	@Override
	public void resetGradients() {
		dW = new Matrix(weights.rows, weights.cols);
		dB = new Matrix(biases.rows, biases.cols);
	}

	@Override
	public void applyGradients(double regTerm, double scale) {
		if(regTerm != 0)
			weights.timesEquals(regTerm);
		weights.plusEquals(dW.timesEquals(-scale));
		biases.plusEquals(dB.timesEquals(-scale));
	}
	
	@Override
	public String toString() {
		return getClass().getSimpleName() + " (input " + weights.cols + " => output " + weights.rows + ")";
	}

	@Override
	public Dimension prepare(Dimension inputDimension) throws IncompatibleLayerException {
		if(inputDimension.depth != 1 || inputDimension.columns != 1)
			throw new IncompatibleLayerException("A " + getClass().getSimpleName()
					+ " can only accept a single column vector, not: " + inputDimension);

		weights = Matrix.random(NEURONS, inputDimension.rows);
		biases = Matrix.random(NEURONS, 1);
		
		return new Dimension(biases.rows, 1, 1);
	}
}
