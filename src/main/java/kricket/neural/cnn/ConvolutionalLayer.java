package kricket.neural.cnn;

import kricket.neural.util.Dimension;
import kricket.neural.util.IncompatibleLayerException;
import kricket.neural.util.SubTensor;
import kricket.neural.util.Tensor;

/**
 * A convolutional layer is composed of one or more kernels. Each kernel is a cube of size
 * (kr, kc, i), where i = the number of input maps. Each kernel is repeated across all
 * valid regions of the input maps. It acts like a fully-connected layer: it has weights
 * for each input it encounters, plus a bias.
 */
public class ConvolutionalLayer implements Layer {
	
	private Tensor[] kernels, dK, oldDK;
	private final int stepX, stepY, kernelRows, kernelCols, numKernels;
	private Tensor lastX, lastY, biases;
	private Tensor dB, oldDB;
	private int outputRows, outputCols;
	private Tensor backAdjust;
	private double momentum;
	
	/**
	 * @param numKernels The number of kernels to use.
	 * @param kernelWidth The width of each kernel.
	 * @param kernelHeight The height of each kernel.
	 * @param colStep The number of columns to skip when applying the kernels.
	 * @param rowStep The number of rows to skip when applying the kernels.
	 */
	public ConvolutionalLayer(int numKernels, int kernelWidth, int kernelHeight, int colStep, int rowStep) {
		this.numKernels = numKernels;
		kernelRows = kernelHeight;
		kernelCols = kernelWidth;
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
		// The output of a single kernel will fill one slice of the output layer.
		// We iterate over the kernels (weights) and apply them to the input maps, + biases, to get
		// the output pixel value.
		
		for(int k=0; k<numKernels; k++) {
			for(int r = 0, or = 0; r <= x.rows - kernelRows; r += stepY, or++) {
				for(int c = 0, oc = 0; c <= x.cols - kernelCols; c += stepX, oc++) {
					SubTensor xs = new SubTensor(x, r, c, 0, kernelRows, kernelCols, x.slices);
					double pixel = xs.innerProduct(kernels[k]) + biases.data[k];
					lastY.set(or, oc, k, pixel);
				}
			}
		}
		
		return lastY;
	}

	@Override
	public Tensor backprop(Tensor deltas) {
		// The idea here is: since each kernel is basically like a single fully-connected neuron,
		// we simply repeat the feedforward loops to pair up the kernels with the sub-regions
		// where they are applied. The backpropagated deltas are the kernels, and the dKs are
		// the original input SubTensors.
		Tensor back = new Tensor(lastX.rows, lastX.cols, lastX.slices);
		
		for(int r=0; r<deltas.rows; r++) for(int c=0; c<deltas.cols; c++) {
			SubTensor xs = new SubTensor(lastX, r*stepY, c*stepX, 0, kernelRows, kernelCols, lastX.slices);
			SubTensor bs = new SubTensor(back, r*stepY, c*stepX, 0, kernelRows, kernelCols, back.slices);
			for(int k=0; k<numKernels; k++) {
				// deltas[r,c,k] = the delta for kernel k applied at x[r*step, c*step]
				double delta = deltas.at(r, c, k);
				dK[k].plusEqualsTimes(xs, delta);
				bs.plusEqualsTimes(kernels[k], delta);
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
		biases.plusEquals(oldDB);
		
		for(int k=0; k<numKernels; k++) {
			if(regTerm != 0)
				kernels[k].timesEquals(regTerm);
			kernels[k].plusEquals(dK[k].timesEquals(scale));
			
			kernels[k].plusEquals(oldDK[k]);
		}
	}

	@Override
	public void resetGradients() {
		for(int i=0; i<numKernels; i++) {
			oldDK[i] = dK[i].timesEquals(momentum);
			dK[i] = new Tensor(kernelRows, kernelCols, kernels[i].slices);
		}
		oldDB = dB.timesEquals(momentum);
		dB = new Tensor(biases.rows, biases.cols, 1);
	}
	
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(getClass().getSimpleName());
		sb.append("\n");
		for(int k=0; k<numKernels; k++) {
			sb.append("Kernel ");
			sb.append(k);
			sb.append("\n");
			for(int s=0; s<kernels[k].slices; s++) {
				sb.append(kernels[k].draw(s));
				sb.append("\n");
			}
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
		outputRows = (inputDimension.rows - kernelRows) / stepY + 1;
		if(outputRows < 1)
			throw new IncompatibleLayerException("Given " + inputDimension + ", we would have " + outputRows + " output rows!");
		
		outputCols = (inputDimension.columns - kernelCols) / stepX + 1;
		if(outputCols < 1)
			throw new IncompatibleLayerException("Given " + inputDimension + ", we would have " + outputCols + " output columns!");
		
		kernels = new Tensor[numKernels];
		dK = new Tensor[numKernels];
		oldDK = new Tensor[numKernels];
		for(int i=0; i<numKernels; i++) {
			kernels[i] = Tensor.random(kernelRows, kernelCols, inputDimension.depth);
			dK[i] = new Tensor(kernelRows, kernelCols, inputDimension.depth);
		}
		biases = Tensor.random(numKernels, 1, 1);
		dB = new Tensor(biases.rows, biases.cols, 1);
		
		lastY = new Tensor(outputRows, outputCols, numKernels);
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
		Tensor fakeKernel = new Tensor(kernelRows, kernelCols, inputDimension.depth);
		for(int i=0; i<fakeKernel.data.length; i++)
			fakeKernel.data[i] = 1;
		
		for(int r=0; r<=inputDimension.rows-kernelRows; r+=stepY) {
			for(int c=0; c<=inputDimension.columns-kernelCols; c+=stepX) {
				SubTensor subBack = new SubTensor(backAdjust, r, c, 0, kernelRows, kernelCols, inputDimension.depth);
				subBack.plusEqualsTimes(fakeKernel, 1);
			}
		}
		
		// One last little check: depending on the configuration, some pixels may never be used!
		// This would leave us with zero entries...which will cause NaNs to appear.
		// The solution here is to set those entries to 1 (the backpropagated value will be 0 anyway).
		for(int i=0; i<backAdjust.data.length; i++) {
			if(backAdjust.data[i] == 0)
				backAdjust.data[i] = 1;
		}
	}
}
