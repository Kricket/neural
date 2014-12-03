package kricket.neural.cnn;

import kricket.neural.util.Dimension;
import kricket.neural.util.IncompatibleLayerException;
import kricket.neural.util.Matrix;

/**
 * A flattening layer simply collapses the input into one giant column vector.
 */
public class FlatteningLayer implements Layer {
	private Matrix[] lastX, lastY, newDelta;
	
	@Override
	public Matrix[] feedForward(Matrix[] x) {
		lastX = x;
		
		if(x.length == 1) {
			lastY[0] = new Matrix(x[0].data);
		} else {
			double[] vector = lastY[0].data;
			for(int i=0; i<x.length; i++) {
				System.arraycopy(x[i].data, 0, vector, i*x[i].data.length, x[i].data.length);
			}
		}
		
		return lastY;
	}

	@Override
	public Matrix[] backprop(Matrix[] deltas) {
		double[] delta = deltas[0].data;
		
		for(int i=0; i<lastX.length; i++) {
			System.arraycopy(delta, i*lastX[i].data.length, newDelta[i].data, 0, newDelta[i].data.length);
		}
		
		return newDelta;
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
		int length = inputDimension.rows * inputDimension.columns * inputDimension.depth;
		
		lastY = new Matrix[] {new Matrix(length, 1)};
		
		newDelta = new Matrix[inputDimension.depth];
		for(int i=0; i<newDelta.length; i++) {
			newDelta[i] = new Matrix(inputDimension.rows, inputDimension.columns);
		}
		
		return new Dimension(length, 1, 1);
	}
}
