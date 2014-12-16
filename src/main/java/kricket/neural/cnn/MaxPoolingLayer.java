package kricket.neural.cnn;

import kricket.neural.util.Dimension;
import kricket.neural.util.IncompatibleLayerException;
import kricket.neural.util.Tensor;

/**
 * A max-pooling layer combines a list of feature maps into a single map by simply selecting
 * the highest value found for each pixel.
 */
public class MaxPoolingLayer implements Layer {
	
	/**
	 * Cache, to avoid re-allocating tensors on each run.
	 */
	private Tensor lastY;
	private int inputDepth;
	/**
	 * Indices of which kernel map had the max value for each pixel.
	 * I.e., if x[3] had the max value for the pixel at (r=2, c=5), then
	 * maxIndices[2][5] = 3
	 */
	private int[][] maxIndices;

	@Override
	public Tensor feedForward(Tensor x) {
		for(int r=0; r<lastY.rows; r++) for(int c=0; c<lastY.cols; c++) {
			double max = Double.MIN_VALUE;
			for(int i=0; i<x.slices; i++) {
				double d = x.at(r, c, i);
				if(d > max) {
					max = d;
					maxIndices[r][c] = i;
				}
			}
			
			lastY.set(r, c, 0, max);
		}
		return lastY;
	}

	@Override
	public Tensor backprop(Tensor deltas) {
		// The easiest way to zero out the tensor...?
		Tensor delta = new Tensor(deltas.rows, deltas.cols, inputDepth);
		
		for(int r=0; r<deltas.rows; r++) for(int c=0; c<deltas.cols; c++) {
			delta.set(r, c, maxIndices[r][c], deltas.at(r, c, 0));
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
		lastY = new Tensor(inputDimension.rows, inputDimension.columns, 1);
		inputDepth = inputDimension.depth;
		maxIndices = new int[inputDimension.rows][];
		for(int i=0; i<maxIndices.length; i++)
			maxIndices[i] = new int[inputDimension.columns];
		
		return new Dimension(inputDimension.rows, inputDimension.columns, 1);
	}

	@Override
	public String toString() {
		return getClass().getSimpleName() + " (input depth: " + inputDepth + ")";
	}
	
	/**
	 * Get the last output that this layer generated.
	 */
	public Tensor getLastOutput() {
		return lastY;
	}
}
