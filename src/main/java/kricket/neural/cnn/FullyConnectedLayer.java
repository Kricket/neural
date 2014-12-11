package kricket.neural.cnn;

import kricket.neural.util.Dimension;
import kricket.neural.util.IncompatibleLayerException;
import kricket.neural.util.Tensor;

/**
 * A fully-connected layer contains a number of neurons. Each neuron's output is a
 * linear function of all the inputs to this layer.
 */
public class FullyConnectedLayer implements Layer {

	/**
	 * The parameters of this Layer.
	 */
	Tensor weights, biases;
	/**
	 * The last input received.
	 */
	Tensor lastX;
	/**
	 * The running total of the calculated gradients of the weights and biases.
	 */
	Tensor dW, dB, oldDW, oldDB;
	
	/**
	 * Temp values, to avoid re-allocating.
	 */
	private Tensor dT_times_x, wT_times_d;
	private final int NEURONS;
	private final double MOMENTUM;
	
	/**
	 * Initialize this layer (with 0 momentum).
	 * @param numNeurons see {@link #FullyConnectedLayer(int, double)}
	 */
	public FullyConnectedLayer(int numNeurons) {
		this(numNeurons, 0);
	}
	
	/**
	 * @param numNeurons The number of neurons in this layer.
	 * @param momentum The momentum factor: how much of the previous gradient we conserve.
	 */
	public FullyConnectedLayer(int numNeurons, double momentum) {
		NEURONS = numNeurons;
		MOMENTUM = momentum;
	}
	
	@Override
	public Tensor feedForward(Tensor x) {
		lastX = x;
		Tensor temp = new Tensor(weights.rows, lastX.cols, 1);
		return weights.times(lastX, temp).plusEquals(biases);
	}

	@Override
	public Tensor backprop(Tensor deltas) {
		/*
		 * backprop is actually two operations:
		 * - calculate the derivatives wrt the weights and biases, and add them to dW and dB
		 * - calculate the derivatives wrt the inputs, and return them
		 */
		dB.plusEquals(deltas);
		dW.plusEquals(deltas.timesTranspose(lastX, dT_times_x));
		
		return weights.transposeTimes(deltas, wT_times_d);
	}

	@Override
	public void resetGradients() {
		oldDW = dW.timesEquals(MOMENTUM);
		oldDB = dB.timesEquals(MOMENTUM);
		dW = new Tensor(weights.rows, weights.cols, 1);
		dB = new Tensor(biases.rows, biases.cols, 1);
	}

	@Override
	public void applyGradients(double regTerm, double scale) {
		if(regTerm != 0)
			weights.timesEquals(regTerm);
		weights.plusEquals(dW.timesEquals(-scale));
		biases.plusEquals(dB.timesEquals(-scale));
		weights.plusEquals(oldDW);
		biases.plusEquals(oldDB);
	}
	
	@Override
	public String toString() {
		return getClass().getSimpleName()
				+ " (input "
				+ weights.cols
				+ " => output "
				+ weights.rows
				+ ", momentum="
				+ MOMENTUM
				+ ")";
	}

	@Override
	public Dimension prepare(Dimension inputDimension) throws IncompatibleLayerException {
		if(inputDimension.depth != 1 || inputDimension.columns != 1)
			throw new IncompatibleLayerException("A " + getClass().getSimpleName()
					+ " can only accept a single column vector, not: " + inputDimension);

		weights = Tensor.random(NEURONS, inputDimension.rows, 1);
		biases = Tensor.random(NEURONS, 1, 1);
		
		dW = new Tensor(weights.rows, weights.cols, 1);
		dB = new Tensor(NEURONS, 1, 1);
		
		dT_times_x = new Tensor(NEURONS, inputDimension.rows, 1);
		wT_times_d = new Tensor(weights.cols, 1, 1);
		
		return new Dimension(biases.rows, 1, 1);
	}
}
