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
	private final int skipCols, skipRows;
	private Tensor lastX, lastY;
	private Tensor dK, dB;
	private int outputRows, outputCols;
	
	public ConvolutionalLayer(int numKernels, int kernelWidth, int kernelHeight, int skipColmuns, int skipRows) {
		kernels = Tensor.random(kernelHeight, kernelWidth, numKernels);
		biases = Tensor.random(numKernels, 1, 1);
		this.skipCols = skipColmuns;
		this.skipRows = skipRows;
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
		for(int s=0; s<x.depth; s++) {
			for(int k=0; k<kernels.depth; k++) {
				for(int r=0; r<outputRows; r++) for(int c=0; c<outputCols; c++) {
					SubTensor subMatrix = x.subMatrix(r*skipRows, c*skipCols, s, kernels.rows, kernels.cols);
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
		Tensor back = new Tensor(lastX.rows, lastX.cols, lastX.depth);
		for(int s=0; s<lastX.depth; s++) {
			for(int r=0; r<deltas.rows; r++) for(int c=0; c<deltas.cols; c++) {
				// deltas[r,c,deltaSlice] = the error of kernel[k] when applied to x[s] at (r*skip, c*skip)
				
				SubTensor subX = lastX.subMatrix(r*skipRows, c*skipCols, s, kernels.rows, kernels.cols);
				SubTensor subBack = back.subMatrix(r*skipRows, c*skipCols, s, kernels.rows, kernels.cols);
				
				for(int k=0; k<kernels.depth; k++) {
					int deltaSlice = s*lastX.depth + k;
					double delta = deltas.at(r, c, deltaSlice);
					dB.data[k] += delta;
					
					// dX/dK = K
					subBack.plusEqualsSliceTimes(kernels, k, delta);
					
					// dK/dX = X
					dK.subMatrix(0, 0, k, dK.rows, dK.cols).plusEqualsTimes(subX, delta);
				}
			}
		}
		/*
		for(int i=0; i<lastX.length; i++) {
			back[i] = new Matrix(lastX[i].rows, lastX[i].cols);
			for(int k=0; k<kernels.length; k++) {
				Matrix delta = deltas[i*lastX.length + k];
				// delta(r,c) = the error for kernel k applied to x[i] at (r*skip, c*skip)
				for(int r=0; r<delta.rows; r++) for(int c=0; c<delta.cols; c++) {
					dB.data[k] += delta.at(r, c);
					
					Matrix xSubMatrix = lastX[i].subMatrix(r*skipRows, c*skipCols, kernels[k].rows, kernels[k].cols);
					dK[k].plusEquals(xSubMatrix.copy().timesEquals(delta.at(r,c)));
					
					Matrix backSubMatrix = back[i].subMatrix(r*skipRows, c*skipCols, kernels[k].rows, kernels[k].cols);
					backSubMatrix.plusEquals(kernels[k]);
				}
			}
		}
		*/
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
	}

	@Override
	public void resetGradients() {
		dB = new Tensor(biases.rows, biases.cols, 1);
		dK = new Tensor(kernels.rows, kernels.cols, kernels.depth);
	}
	
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(getClass().getSimpleName());
		sb.append("\n");
		for(int k=0; k<kernels.depth; k++) {
			sb.append("Kernel ");
			sb.append(k);
			sb.append("\n");
			sb.append(kernels.draw(k));
		}
		sb.append("skip rows=");
		sb.append(skipRows);
		sb.append(" cols=");
		sb.append(skipCols);
		return sb.toString();
	}

	@Override
	public Dimension prepare(Dimension inputDimension) throws IncompatibleLayerException {
		outputRows = (inputDimension.rows - kernels.rows) / skipRows + 1;
		if(outputRows < 1)
			throw new IncompatibleLayerException("Given " + inputDimension + ", we would have " + outputRows + " output rows!");
		
		outputCols = (inputDimension.columns - kernels.cols) / skipCols + 1;
		if(outputCols < 1)
			throw new IncompatibleLayerException("Given " + inputDimension + ", we would have " + outputCols + " output columns!");
		
		lastY = new Tensor(outputRows, outputCols, inputDimension.depth * kernels.depth);
		
		return lastY.getDimension();
	}
}
