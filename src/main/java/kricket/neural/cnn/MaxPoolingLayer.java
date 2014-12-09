package kricket.neural.cnn;

import kricket.neural.util.Dimension;
import kricket.neural.util.IncompatibleLayerException;
import kricket.neural.util.Matrix;

/**
 * A max-pooling layer combines a list of feature maps into a single map by simply selecting
 * the highest value found for each pixel.
 */
public class MaxPoolingLayer implements Layer {
	
	/**
	 * Cache, to avoid re-allocating matrices on each run.
	 */
	private Matrix[] lastY, delta;
	/**
	 * Indices of which kernel map had the max value for each pixel.
	 * I.e., if x[3] had the max value for the pixel at (r=2, c=5), then
	 * maxIndices[2][5] = 3
	 */
	private int[][] maxIndices;

	@Override
	public Matrix[] feedForward(Matrix[] x) {
		for(int r=0; r<lastY[0].rows; r++) for(int c=0; c<lastY[0].cols; c++) {
			double max = Double.MIN_VALUE;
			for(int i=0; i<x.length; i++) {
				double d = x[i].at(r, c);
				if(d > max) {
					max = d;
					maxIndices[r][c] = i;
				}
			}
			
			lastY[0].set(r, c, max);
		}
		return lastY;
	}

	@Override
	public Matrix[] backprop(Matrix[] deltas) {
		// The easiest way to zero out the matrices...?
		for(int i=0; i<delta.length; i++) {
			delta[i] = new Matrix(deltas[0].rows, deltas[0].cols);
		}
		
		for(int r=0; r<deltas[0].rows; r++) for(int c=0; c<deltas[0].cols; c++) {
			delta[maxIndices[r][c]].set(r, c, deltas[0].at(r, c));
		}
		
		return delta;
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
		lastY = new Matrix[] {new Matrix(inputDimension.rows, inputDimension.columns)};
		delta = new Matrix[inputDimension.depth];
		maxIndices = new int[inputDimension.rows][];
		for(int i=0; i<maxIndices.length; i++)
			maxIndices[i] = new int[inputDimension.columns];
		
		return new Dimension(inputDimension.rows, inputDimension.columns, 1);
	}

}
