package kricket.neural.cnn;

import kricket.neural.util.Matrix;

/**
 * A flattening layer simply collapses the input into one giant column vector.
 */
public class FlatteningLayer implements Layer {
	private Matrix[] lastX;
	
	@Override
	public Matrix[] feedForward(Matrix[] x) {
		lastX = x;
		
		double[] vector;
		if(x.length == 1) {
			vector = x[0].data;
		} else {
			vector = new double[x.length * x[0].data.length];
			for(int i=0; i<x.length; i++) {
				System.arraycopy(x[i].data, 0, vector, i*x[i].data.length, x[i].data.length);
			}
		}
		
		return new Matrix[]{new Matrix(vector)};
	}

	@Override
	public Matrix[] backprop(Matrix[] deltas) {
		double[] delta = deltas[0].data;
		Matrix[] back = new Matrix[lastX.length];
		
		for(int i=0; i<lastX.length; i++) {
			back[i] = new Matrix(lastX[i].rows, lastX[i].cols);
			System.arraycopy(delta, i*lastX[i].data.length, back[i].data, 0, back[i].data.length);
		}
		
		return back;
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
}
