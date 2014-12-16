package kricket.neural.cnn;

import kricket.neural.util.Dimension;
import kricket.neural.util.IncompatibleLayerException;
import kricket.neural.util.SubTensor;
import kricket.neural.util.Tensor;

/**
 * A convolutional layer is composed of one or more kernels. Each kernel is a small, rectangular
 * pattern that acts like a single neuron of a fully-connected layer, but is applied repeatedly
 * for a single input feature map.
 */
public class ConvolutionalLayer implements Layer {
	
	private final Tensor kernels, biases;
	private final int stepX, stepY;
	private Tensor lastX, lastY;
	private Tensor dK, dB, oldDK, oldDB;
	private int outputRows, outputCols;
	private Tensor backAdjust;
	private double momentum;
	
	/**
	 * 
	 * @param numKernels The number of kernels to use.
	 * @param kernelWidth The width of each kernel.
	 * @param kernelHeight The height of each kernel.
	 * @param colStep The number of columns to skip when applying the kernels.
	 * @param rowStep The number of rows to skip when applying the kernels.
	 */
	public ConvolutionalLayer(int numKernels, int kernelWidth, int kernelHeight, int colStep, int rowStep) {
		kernels = Tensor.random(kernelHeight, kernelWidth, numKernels);
		biases = Tensor.random(numKernels, 1, 1);
		stepX = colStep;
		stepY = rowStep;
	}
	
	/**
	 * Add a momentum factor to the gradients of the kernel weights and biases.
	 * @param p
	 * @return
	 */
	public ConvolutionalLayer withMomentum(double p) {
		momentum = p;
		return this;
	}
	
	/**
	 * Get the number of rows in an output feature map.
	 * @return
	 */
	public int getOutputRows() {
		return outputRows;
	}
	
	/**
	 * Get the number of columns in an output feature map.
	 * @return
	 */
	public int getOutputColumns() {
		return outputCols;
	}

	@Override
	public Tensor feedForward(Tensor x) {
		lastX = x;
		
		// Each kernel is basically like the weights of a single neuron of a fully-connected layer.
		// We iterate over the input maps, and apply the kernels (weights) to the sub-maps, + biases,
		// to get the output pixel value.
		int ySlice = 0;
		for(int s=0; s<x.slices; s++) {
			for(int k=0; k<kernels.slices; k++) {
				for(int r=0; r<outputRows; r++) for(int c=0; c<outputCols; c++) {
					SubTensor subMatrix = x.subMatrix(r*stepY, c*stepX, s, kernels.rows, kernels.cols);
					double weighted = subMatrix.innerProduct(kernels, k);
					lastY.set(r, c, ySlice, weighted + biases.data[k]);
				}
				ySlice++;
			}
		}
		
		return lastY;
	}

	@Override
	public Tensor backprop(Tensor deltas) {
		// The idea here is: since each kernel is basically like a single fully-connected neuron,
		// we simply repeat the feedforward loops to pair up the kernels with the sub-images
		// where they are applied. The backpropagated deltas are the kernels, and the dKs are
		// the original sub-matrices.
		Tensor back = new Tensor(lastX.rows, lastX.cols, lastX.slices);
		for(int s=0; s<lastX.slices; s++) {
			for(int r=0; r<deltas.rows; r++) for(int c=0; c<deltas.cols; c++) {
				// deltas[r,c,deltaSlice] = the error of kernel[k] when applied to x[s] at (r*skip, c*skip)
				
				SubTensor subX = lastX.subMatrix(r*stepY, c*stepX, s, kernels.rows, kernels.cols);
				SubTensor subBack = back.subMatrix(r*stepY, c*stepX, s, kernels.rows, kernels.cols);
				
				for(int k=0; k<kernels.slices; k++) {
					int deltaSlice = s*lastX.slices + k;
					double delta = deltas.at(r, c, deltaSlice);
					dB.data[k] += delta;
					
					// dX/dK = K
					subBack.plusEqualsSliceTimes(kernels, k, delta);
					
					// dK/dX = X
					dK.subMatrix(0, 0, k, dK.rows, dK.cols).plusEqualsTimes(subX, delta);
				}
			}
		}
		
		// Now, apply the adjustment
		for(int i=0; i<back.data.length; i++) {
			back.data[i] /= backAdjust.data[i];
		}
		return back;
	}

	@Override
	public void applyGradients(double regTerm, double scale) {
		// Since each kernel was repeated r*c times, we have to reduce the gradients by that much
		scale = -scale / (outputRows * outputCols);
		biases.plusEquals(dB.timesEquals(scale));
		
		if(regTerm != 0)
			kernels.timesEquals(regTerm);
		kernels.plusEquals(dK.timesEquals(scale));
		
		kernels.plusEquals(oldDK);
		biases.plusEquals(oldDB);
	}

	@Override
	public void resetGradients() {
		oldDK = dK.timesEquals(momentum);
		oldDB = dB.timesEquals(momentum);
		dB = new Tensor(biases.rows, biases.cols, 1);
		dK = new Tensor(kernels.rows, kernels.cols, kernels.slices);
	}
	
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(getClass().getSimpleName());
		sb.append("\n");
		for(int k=0; k<kernels.slices; k++) {
			sb.append("Kernel ");
			sb.append(k);
			sb.append("\n");
			sb.append(kernels.draw(k));
		}
		sb.append("skip rows=");
		sb.append(stepY);
		sb.append(" cols=");
		sb.append(stepX);
		sb.append(" ==> ");
		sb.append(outputRows);
		sb.append(" x ");
		sb.append(outputCols);
		sb.append(" output");
		return sb.toString();
	}

	@Override
	public Dimension prepare(Dimension inputDimension) throws IncompatibleLayerException {
		outputRows = (inputDimension.rows - kernels.rows) / stepY + 1;
		if(outputRows < 1)
			throw new IncompatibleLayerException("Given " + inputDimension + ", we would have " + outputRows + " output rows!");
		
		outputCols = (inputDimension.columns - kernels.cols) / stepX + 1;
		if(outputCols < 1)
			throw new IncompatibleLayerException("Given " + inputDimension + ", we would have " + outputCols + " output columns!");
		
		dB = new Tensor(biases.rows, biases.cols, 1);
		dK = new Tensor(kernels.rows, kernels.cols, kernels.slices);
		
		lastY = new Tensor(outputRows, outputCols, inputDimension.depth * kernels.slices);
		setupBackAdjust(inputDimension);
		
		return lastY.getDimension();
	}

	/**
	 * So, there's a little problem with what we backpropagate. Depending on
	 * the step and kernel sizes, kernels can overlap. For example, for a kernel
	 * of width 3, with stepX=1:
	 * - the first and last pixels on each row will be used just once
	 * - the second and second-to-last pixels on each row will be used twice
	 * - all other pixels on the row will be used 3 times
	 * The result of all this is that, when we backpropagate, some pixels will
	 * carry more data than others. To even them out, we use the Tensor created
	 * in this method to scale them appropriately.
	 * @param inputDimension 
	 */
	private void setupBackAdjust(Dimension inputDimension) {
		backAdjust = new Tensor(inputDimension);
		Tensor fakeKernel = new Tensor(kernels.rows, kernels.cols, 1);
		for(int i=0; i<fakeKernel.data.length; i++)
			fakeKernel.data[i] = 1;
		
		// For starters, just do the first slice...
		for(int r=0; r<=inputDimension.rows-kernels.rows; r+=stepY) {
			for(int c=0; c<=inputDimension.columns-kernels.cols; c+=stepX) {
				SubTensor subBack = backAdjust.subMatrix(r, c, 0, kernels.rows, kernels.cols);
				subBack.plusEqualsSliceTimes(fakeKernel, 0, 1);
			}
		}
		
		// ...then, copy it to all the other slices.
		for(int d=0; d<backAdjust.slices; d++) {
			int idx = backAdjust.index(0, 0, d);
			System.arraycopy(backAdjust.data, 0, backAdjust.data, idx, backAdjust.rows*backAdjust.cols);
		}
	}
}
