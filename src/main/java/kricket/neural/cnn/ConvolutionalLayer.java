package kricket.neural.cnn;

import kricket.neural.util.Dimension;
import kricket.neural.util.IncompatibleLayerException;
import kricket.neural.util.Matrix;

public class ConvolutionalLayer extends Layer {
	
	private final Matrix[] kernels;
	private final Matrix biases;
	private final int skipCols, skipRows;
	private Matrix[] lastActivation, lastZ;
	private Matrix[] nabla_Ck;
	private Matrix nabla_Cb;
	
	public ConvolutionalLayer(int numKernels, int kernelWidth, int kernelHeight, int skipColmuns, int skipRows) {
		kernels = new Matrix[numKernels];
		biases = Matrix.random(numKernels, 1);
		for(int i=0; i<numKernels; i++) {
			kernels[i] = Matrix.random(kernelHeight, kernelWidth);
		}
		this.skipCols = skipColmuns;
		this.skipRows = skipRows;
		
		nabla_Ck = new Matrix[numKernels];
	}
	
	@Override
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
	public Matrix[] feedForward(Matrix[] featureMaps) {
		lastZ = new Matrix[featureMaps.length * kernels.length];
		
		int resultIndex = 0;
		for(Matrix inputMap : featureMaps) {
			int rows = getOutputRows(inputMap.rows), cols = getOutputColumns(inputMap.cols);
			for(int k=0; k<kernels.length; k++) {
				Matrix outMap = lastZ[resultIndex++] = new Matrix(rows, cols);
				for(int r=0; r<rows; r++) for(int c=0; c<cols; c++) {
					outMap.set(r, c, inputMap.subMatrixDot(r*skipRows, c*skipCols, kernels[k]) + biases.data[k]);
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
		throw new UnsupportedOperationException("TODO");
	}

	@Override
	public void calcGradients(Matrix[] prevActivations, Matrix[] deltas) {
		// We should have 1 delta Matrix for each output feature map.
		deltas = unflatten(deltas, prevActivations);
		
		if(deltas.length != prevActivations.length * kernels.length)
			throw new IllegalArgumentException("WTF? We have " + deltas.length + " deltas (output feature maps), "
					+ kernels.length + " kernels, and " + prevActivations.length + " previous feature maps!");
		
		for(int m = 0; m < prevActivations.length; m++) {
			for(int k=0; k<kernels.length; k++) {
				// deltas[m * k] = delta for the map built with kernel k and input map m
				Matrix delta = deltas[m*k];
				for(int dr=0; dr<delta.rows; dr++) {
					for(int dc=0; dc<delta.cols; dc++) {
						// The values here should be reduced by the number of times the kernel is repeated (i.e., the
						// size of delta). As an optimization, we do it in applyGradients, instead.
						nabla_Cb.data[k] += delta.at(dr, dc);
						nabla_Ck[k].plusEqualsSubMatrix(prevActivations[m], dr*skipRows, dc*skipCols, delta.at(dr, dc));
					}
				}
			}
		}
	}

	/**
	 * Inverse of the "flatten" operation. I.e.: in case we happened to receive a single column vector
	 * as our "deltas", we need to explode it into the correct array of deltas of each output feature map.
	 * @param deltas
	 * @param inputs
	 * @return
	 */
	public Matrix[] unflatten(Matrix[] deltas, Matrix[] inputs) {
		if(deltas.length > 1) {
			if(deltas.length != inputs.length * kernels.length)
				throw new IllegalArgumentException("Unable to unflatten " + deltas.length
						+ " deltas, when there are " + inputs.length + " input maps and "
						+ kernels.length + " kernels");
			return deltas;
		}
		
		Matrix[] result = new Matrix[inputs.length * kernels.length];
		int offset = 0;
		
		for(int i=0; i<inputs.length; i++) {
			int outputRows = getOutputRows(inputs[i].rows),
					outputColumns = getOutputColumns(inputs[i].cols);
			
			for(int k=0; k<kernels.length; k++) {
				double[] delta_ik = new double[outputRows * outputColumns];
				System.arraycopy(deltas[0].data, offset, delta_ik, 0, delta_ik.length);
				result[i*kernels.length + k] = new Matrix(outputRows, outputColumns, delta_ik);
				offset += delta_ik.length;
			}
		}
		
		return result;
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
		int scaleDown = getOutputColumns(lastActivation[0].cols) * getOutputRows(lastActivation[0].rows);
		double factor = -eta / (batchSize*scaleDown);
		biases.plusEquals(nabla_Cb.timesEquals(factor));
		
		for(int k=0; k<kernels.length; k++) {
			if(regTerm != 0)
				kernels[k].timesEquals(regTerm);
			kernels[k].plusEquals(nabla_Ck[k].timesEquals(factor));
		}
	}

	@Override
	public void resetGradients() {
		nabla_Cb = new Matrix(biases.rows, biases.cols);
		for(int i=0; i<nabla_Ck.length; i++)
			nabla_Ck[i] = new Matrix(kernels[i].rows, kernels[i].cols);
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
