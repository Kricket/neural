package kricket.neural.cnn;

import kricket.neural.util.Dimension;
import kricket.neural.util.IncompatibleLayerException;
import kricket.neural.util.Tensor;

/**
 * A flattening layer simply collapses the input into one giant column vector.
 */
public class FlatteningLayer implements Layer {
	private Dimension inputDimension;
	
	@Override
	public Tensor feedForward(Tensor x) {
		return new Tensor(x.data);
	}

	@Override
	public Tensor backprop(Tensor deltas) {
		return new Tensor(inputDimension, deltas.data);
	}

	@Override
	public void applyGradients(double regTerm, double scale) {
		// Nothing to do
	}

	@Override
	public void resetGradients() {
		// Nothing to do
	}

	@Override
	public String toString() {
		return getClass().getSimpleName();
	}

	@Override
	public Dimension prepare(Dimension inputDimension) throws IncompatibleLayerException {
		this.inputDimension = inputDimension;
		int length = inputDimension.rows * inputDimension.columns * inputDimension.depth;
		return new Dimension(length, 1, 1);
	}
}
