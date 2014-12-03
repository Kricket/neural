package kricket.neural.cnn;

import kricket.neural.util.Dimension;
import kricket.neural.util.IncompatibleLayerException;
import kricket.neural.util.Matrix;

/**
 * A convolutional layer is composed of one or more kernels. Each kernel is a small, rectangular
 * pattern that acts like a single neuron of a fully-connected layer, but is applied repeatedly
 * for a single input feature map.
 */
public class ConvolutionalLayer implements Layer {
	
	private final Matrix[] kernels;
	private final Matrix biases;
	private final int skipCols, skipRows;
	private Matrix[] lastX;
	private Matrix[] dK;
	private Matrix dB;
	
	public ConvolutionalLayer(int numKernels, int kernelWidth, int kernelHeight, int skipColmuns, int skipRows) {
		kernels = new Matrix[numKernels];
		biases = Matrix.random(numKernels, 1);
		for(int i=0; i<numKernels; i++) {
			kernels[i] = Matrix.random(kernelHeight, kernelWidth);
		}
		this.skipCols = skipColmuns;
		this.skipRows = skipRows;
		
		dK = new Matrix[numKernels];
	}
	
	public Dimension getOutputDimension(Dimension inputDimension) throws IncompatibleLayerException {
		int rows = getOutputRows(inputDimension.rows), cols = getOutputColumns(inputDimension.columns);
		if(rows < 1 || cols < 1 || inputDimension.depth < 1)
			throw new IncompatibleLayerException(inputDimension, this);
		return new Dimension(rows, cols, inputDimension.depth * kernels.length);
	}
	
	/**
	 * Get the number of rows in an output feature map, if the input feature map has
	 * the given number of rows.
	 * @param inputRows
	 * @return
	 */
	public int getOutputRows(int inputRows) {
		return (inputRows - kernels[0].rows) / skipRows + 1;
	}
	
	/**
	 * Get the number of columns in an output feature map, if the input feature map has
	 * the given number of columns.
	 * @param inputCols
	 * @return
	 */
	public int getOutputColumns(int inputCols) {
		return (inputCols - kernels[0].cols) / skipCols + 1;
	}

	@Override
	public Matrix[] feedForward(Matrix[] x) {
		lastX = x;
		Matrix[] y = new Matrix[x.length * kernels.length]; 
		
		int resultIndex = 0;
		for(Matrix inputMap : x) {
			int rows = getOutputRows(inputMap.rows), cols = getOutputColumns(inputMap.cols);
			for(int k=0; k<kernels.length; k++) {
				Matrix outMap = y[resultIndex++] = new Matrix(rows, cols);
				for(int r=0; r<rows; r++) for(int c=0; c<cols; c++) {
					Matrix subMatrix = inputMap.subMatrix(r*skipRows, c*skipCols, kernels[k].rows, kernels[k].cols);
					outMap.set(r, c, subMatrix.dot(kernels[k]) + biases.data[k]);
				}
			}
		}
		
		return y;
	}

	@Override
	public Matrix[] backprop(Matrix[] deltas) {
		if(deltas.length != kernels.length * lastX.length)
			throw new IllegalArgumentException("Should have " + (kernels.length * lastX.length)
					+ " deltas, but there are " + deltas.length);
		
		// The idea here is: since each kernel is basically like a single fully-connected neuron,
		// we simply repeat the feedforward loops to pair up the kernels with the sub-images
		// where they are applied.
		Matrix[] back = new Matrix[lastX.length];
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
		
		return back;
	}

	@Override
	public void applyGradients(double regTerm, double scale) {
		biases.plusEquals(dB.timesEquals(-scale));
		
		for(int k=0; k<kernels.length; k++) {
			if(regTerm != 0)
				kernels[k].timesEquals(regTerm);
			kernels[k].plusEquals(dK[k].timesEquals(-scale));
		}
	}

	@Override
	public void resetGradients() {
		dB = new Matrix(biases.rows, biases.cols);
		for(int i=0; i<dK.length; i++)
			dK[i] = new Matrix(kernels[i].rows, kernels[i].cols);
	}
	
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(getClass().getSimpleName());
		sb.append("\n");
		for(int k=0; k<kernels.length; k++) {
			sb.append("Kernel ");
			sb.append(k);
			sb.append("\n");
			sb.append(kernels[k].draw());
		}
		sb.append("skip rows=");
		sb.append(skipRows);
		sb.append(" cols=");
		sb.append(skipCols);
		return sb.toString();
	}
}
