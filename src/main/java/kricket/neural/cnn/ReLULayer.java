package kricket.neural.cnn;

import kricket.neural.util.Dimension;
import kricket.neural.util.IncompatibleLayerException;
import kricket.neural.util.Tensor;

/**
 * A Rectified Linear Unit simply outputs max(0,x) for each input x.
 */
public class ReLULayer implements Layer {
	
	private Tensor lastX, lastY;

	private double rectify(double x) {
		return (x > 0 ? x : 0);
	}
	
	@Override
	public Tensor feedForward(Tensor x) {
		lastX = x;
		for(int i=0; i<x.data.length; i++)
			lastY.data[i] = rectify(x.data[i]);
		
		return lastY;
	}

	@Override
	public Tensor backprop(Tensor deltas) {
		// The gradient here is simple: the only parts that get backpropagated
		// are the ones that correspond to positive inputs.
		for(int i=0; i<lastX.data.length; i++) {
			if(lastX.data[i] <= 0)
				deltas.data[i] = 0;
		}
		
		return deltas;
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
	public Dimension prepare(Dimension inputDimension) throws IncompatibleLayerException {
		lastY = new Tensor(inputDimension);
		return inputDimension;
	}

	@Override
	public String toString() {
		return getClass().getSimpleName();
	}
}
