package kricket.neural.cnn;

import kricket.neural.util.Dimension;
import kricket.neural.util.IncompatibleLayerException;
import kricket.neural.util.Matrix;

public class ConvolutionalLayer extends Layer {
	
	private final Matrix[] kernels;
	private final double[] biases;
	private final int skipCols, skipRows;
	private Matrix[] lastActivation, lastZ;
	
	public ConvolutionalLayer(int numKernels, int kernelWidth, int kernelHeight, int skipColmuns, int skipRows) {
		kernels = new Matrix[numKernels];
		biases = new double[numKernels];
		for(int i=0; i<numKernels; i++) {
			kernels[i] = Matrix.random(kernelHeight, kernelWidth);
			biases[i] = Math.random() - Math.random();
		}
		this.skipCols = skipColmuns;
		this.skipRows = skipRows;
	}

	@Override
	public Dimension getOutputDimension(Dimension inputDimension) throws IncompatibleLayerException {
		int rows = getOutputRows(inputDimension.rows), cols = getOutputColumns(inputDimension.columns);
		if(rows < 1 || cols < 1 || inputDimension.depth < 1)
			throw new IncompatibleLayerException(inputDimension, this);
		return new Dimension(rows, cols, inputDimension.depth * kernels.length);
	}
	
	private int getOutputRows(int inputRows) {
		return (inputRows - kernels[0].rows) / skipRows + 1;
	}
	
	private int getOutputColumns(int inputCols) {
		return (inputCols - kernels[0].cols) / skipCols + 1;
	}

	@Override
	public Matrix[] feedForward(Matrix[] featureMaps) {
		lastZ = new Matrix[featureMaps.length * kernels.length];
		
		int resultIndex = 0;
		for(Matrix inputMap : featureMaps) {
			int rows = getOutputRows(inputMap.rows), cols = getOutputColumns(inputMap.cols);
			for(int k=0; k<kernels.length; k++) {
				Matrix outMap = lastZ[resultIndex++] = new Matrix(rows, cols);
				for(int r=0; r<rows; r++) for(int c=0; c<cols; c++) {
					outMap.set(r, c, inputMap.subMatrixDot(r*skipRows, c*skipCols, kernels[k]) + biases[k]);
				}
			}
		}
		
		lastActivation = sigma(lastZ);
		return lastActivation;
	}

	private Matrix[] sigma(Matrix[] zs) {
		Matrix[] m = new Matrix[zs.length];
		for(int i=0; i<m.length; i++)
			m[i] = sigma(zs[i].copy());
		return m;
	}

	@Override
	public Matrix[] backprop(Matrix[] prevZ, Matrix[] deltas) {
		return null;
	}

	@Override
	public void calcGradients(Matrix[] prevActivations, Matrix[] deltas) {
	}

	@Override
	public Matrix[] lastActivation() {
		return lastActivation;
	}

	@Override
	public Matrix[] lastZ() {
		return lastZ;
	}

	@Override
	public void applyGradients(double regTerm, double eta, int batchSize) {
	}

	@Override
	public void resetGradients() {
	}
	
	public String toString() {
		return getClass().getSimpleName()
				+ kernels.length + " kernels (r=" + kernels[0].rows + ", c=" + kernels[0].cols
				+ ") skip (r=" + skipRows + ", c=" + skipCols + ")";
	}
}
