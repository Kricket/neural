package kricket.neural.nn;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import kricket.neural.NNBase;
import kricket.neural.util.Datum;
import kricket.neural.util.Matrix;
import kricket.neural.util.NNOptions;

/**
 * A simple neural network.
 */
public class NN extends NNBase {
	/**
	 * Used to accumulate the changes to apply to the weights and biases, calculated
	 * during backpropagation.
	 */
	private static class NablaC {
		public Matrix[] b, w;
		public NablaC(Matrix[] _b, Matrix [] _w) {
			b = _b;
			w = _w;
		}
		public void plusEquals(NablaC other) {
			for(int i=0; i<w.length; i++) {
				w[i].plusEquals(other.w[i]);
				b[i].plusEquals(other.b[i]);
			}
		}
	}
	
	/**
	 * Li = weights[i] represents all the SNs of the (i+1)th layer. The trick is: if we represent
	 * the input as a column vector with k rows, then Li must have k columns. The number of
	 * SNs on layer (i+1) is equal to the number of rows in Li. The "raw" output of layer i
	 * (before the sigma function) is therefore
	 * (input of layer i-2) * (Li) + Bi.
	 */
	private Matrix[] weights, biases;
	
	/**
	 * Total number of layers, including the input layer.
	 */
	private final int NUM_LAYERS;

	/**
	 * Create a new NN with the default options.
	 * @param layers The number of inputs on each layer (INCLUDING the initial layer!).
	 */
	public NN(int ...layers) {
		this(new NNOptions(), layers);
	}
	
	/**
	 * Create a new NN.
	 * @param opts Options to use.
	 * @param layers The number of inputs on each layer (INCLUDING the initial layer!).
	 */
	public NN(NNOptions opts, int ...layers) {
		super(opts);
		
		if(layers.length < 2)
			throw new IllegalArgumentException("You must specify at least two layers (including the initial input layer)");
		
		NUM_LAYERS = layers.length;
		weights = new Matrix[NUM_LAYERS-1];
		biases = new Matrix[NUM_LAYERS-1];
		
		for(int i=1; i<NUM_LAYERS; i++) {
			// Number of columns of the input; one row for each SN in the output
			weights[i-1] = Matrix.random(layers[i], layers[i-1]);
			biases[i-1] = Matrix.random(layers[i], 1);
		}
	}
	
	/**
	 * Get the NN's output from the given input. Note that, since this is NOT a convolutional
	 * network, the given input is flattened into a single vector!
	 * @param input
	 * @return
	 */
	public Matrix feedForward(Matrix input) {
		Matrix current = new Matrix(input.data);
		for(int i=0; i<(NUM_LAYERS-1); i++) {
			current = weights[i].times(current).plusEquals(biases[i]);
			sigma(current);
		}
		
		return current;
	}
	
	/**
	 * @param x The input (will be flattened into a vector)
	 * @param y The "correct" output vector
	 * @return The gradient vector of the cost function, for all the weights and biases.
	 */
	private NablaC backprop(Matrix x, Matrix y) {
		Matrix[] nabla_Cb = new Matrix[NUM_LAYERS - 1];
		Matrix[] nabla_Cw = new Matrix[NUM_LAYERS - 1];
		// The "z" vectors are the non-activated outputs of the SNs of each layer. The first one is null
		// for convenience. Note that they will get modified in-place when we call dSigma! 
		Matrix[] zs = new Matrix[NUM_LAYERS];
		// The activations are the z vectors, with sigma applied. The first one is just the input.
		Matrix[] activations = new Matrix[NUM_LAYERS];
		activations[0] = new Matrix(x.data);
		
		// Step forward through the network, saving the z and sigma(z) on each layer.
		for(int i=1; i<NUM_LAYERS; i++) {
			zs[i] = weights[i-1].times(activations[i-1]).plusEquals(biases[i-1]);
			activations[i] = sigma(zs[i].copy());
		}

		// Now, start working backwards. We have to start manually with the last layer...
		Matrix delta = activations[NUM_LAYERS-1]
				.minus(y)
				// WITH the following line, we use quadratic cost.
				// WITHOUT, we're using cross-entropy (which should learn faster).
				//.dotTimesEquals(dSigma(zs[NUM_LAYERS-1]))
				;
		nabla_Cb[NUM_LAYERS-2] = delta;
		nabla_Cw[NUM_LAYERS-2] = delta.timesTranspose(activations[NUM_LAYERS-2]);
		
		// ...and now walk backwards through the remaining layers.
		for(int layer = NUM_LAYERS-2; layer > 0; layer--) {
			// Transform delta from (layer+1) to (layer)
			delta = weights[layer]
					.transposeTimes(delta)
					.dotTimesEquals(dSigma(zs[layer]));
			
			nabla_Cb[layer-1] = delta;
			nabla_Cw[layer-1] = delta.timesTranspose(activations[layer-1]);
		}
		
		return new NablaC(nabla_Cb, nabla_Cw);
	}
	
	/**
	 * The sigma function, for smoothing.
	 * @param z
	 * @return
	 */
	private double sigma(double z) {
		return 1. / (Math.expm1(-z) + 2.);
	}
	
	/**
	 * Apply the sigma function to every element of the given Matrix. Note that this will CHANGE the matrix!
	 */
	private Matrix sigma(Matrix m) {
		for(int i=0; i<m.data.length; i++)
			m.data[i] = sigma(m.data[i]);
		return m;
	}
	
	/**
	 * Derivative of the sigma function.
	 * @param z
	 * @return
	 */
	private double dSigma(double z) {
		double sigma = sigma(z);
		return sigma * (1 - sigma);
	}
	
	/**
	 * Apply the derivative of the sigma function to every element of the given Matrix. Note that this will CHANGE the matrix!
	 */
	private Matrix dSigma(Matrix m) {
		for(int i=0; i<m.data.length; i++)
			m.data[i] = dSigma(m.data[i]);
		return m;
	}
	
	@Override
	protected void runBatch(List<? extends Datum> batch, double regTerm, double eta) {
		NablaC nabla = null;
		for(Datum dat : batch) {
			NablaC backprop = backprop(dat.getData(), dat.getAnswer());
			if(nabla == null)
				nabla = backprop;
			else
				nabla.plusEquals(backprop);
		}
		
		// Update w and b
		for(int i=0; i<NUM_LAYERS-1; i++) {
			if(regTerm != 0)
				weights[i].timesEquals(regTerm);
			weights[i].plusEquals(nabla.w[i].timesEquals(-eta / batch.size()));
			biases[i].plusEquals(nabla.b[i].timesEquals(-eta / batch.size()));
		}
	}
	
	/**
	 * Calculate the error for the given training set.
	 * @param data
	 * @return The percent correct.
	 */
	public double calc_error(List<? extends Datum> data) {
		double numCorrect = 0;
		for(Datum dat : data) {
			Matrix result = feedForward(dat.getData());
			if(isCorrect(result.data, dat.getAnswer().data)) {
				numCorrect++;
			} else if(options.logIncorrectAnswers) {
				options.log.info("Got this one wrong:\n" + dat);
				for(double d : result.data)
					options.log.info(String.format("%.3f", d));
			}
		}
		
		double correct = numCorrect / data.size();
		options.log.info(String.format("-------------------> Percent correct: %.3f", correct*100));
		return correct;
	}
	
	
	//--- Load and save -----------------------------------------------------//
	
	
	/**
	 * Serialize this NN (weights and biases only) to the given file.
	 * @param filename
	 * @throws IOException
	 */
	public void save(String filename) throws IOException {
		FileOutputStream fs = new FileOutputStream(filename);
		DataOutputStream dos = new DataOutputStream(fs);
		
		try {
			dos.writeInt(NUM_LAYERS);
			dos.writeInt(weights[0].cols);
			for(int i=0; i<weights.length; i++) {
				dos.writeInt(weights[i].rows);
				writeMatrix(dos, weights[i]);
				writeMatrix(dos, biases[i]);
			}
		} finally {
			dos.close();
		}
	}

	private void writeMatrix(DataOutputStream dos, Matrix m) throws IOException {
		for(double val : m.data) {
			dos.writeDouble(val);
		}
	}
	
	/**
	 * Load a NN that was previously saved by {@link #save(String)}.
	 * @param filename
	 * @return
	 * @throws IOException
	 */
	public static NN load(String filename) throws IOException {
		FileInputStream fs = new FileInputStream(filename);
		DataInputStream dis = new DataInputStream(fs);
		
		try {
			int nLayers = dis.readInt();
			NN nn = new NN(nLayers);
			
			int prevLayerSize = dis.readInt();
			for(int i=1; i<nLayers; i++) {
				int layerISize = dis.readInt();
				nn.weights[i-1] = readMatrix(dis, layerISize, prevLayerSize);
				nn.biases[i-1] = readMatrix(dis, layerISize, 1);
				prevLayerSize = layerISize;
			}
			
			return nn;
		} finally {
			dis.close();
		}
	}
	
	private static Matrix readMatrix(DataInputStream dis, int rows, int cols) throws IOException {
		Matrix m = new Matrix(rows, cols);
		for(int i=0; i<m.data.length; i++) {
			m.data[i] = dis.readDouble();
		}
		return m;
	}

	private NN(int nLayers) {
		super(new NNOptions());
		NUM_LAYERS = nLayers;
		weights = new Matrix[NUM_LAYERS-1];
		biases = new Matrix[NUM_LAYERS-1];
	}
	
	
	//--- DROPOUT! ----------------------------------------------------------//
	
	
	private Set<Integer> removedRows;
	private int dropoutLayer;
	private Matrix dropoutWeight, dropoutBias, dropoutWeightPlusOne;
	
	/**
	 * Perform dropout on the given layer (MUST be a HIDDEN layer!).
	 * @param layer
	 */
	public void dropout(int layer) {
		if(removedRows != null)
			throw new UnsupportedOperationException("Oh shit, TWO dropouts???");
		if(layer < 1 || layer >= weights.length)
			throw new UnsupportedOperationException("Dropout only makes sense for hidden layers, not layer " + layer);
		dropoutLayer = layer-1; // Offset, since our list of weights starts with layer 1
		
		// Randomly remove half the rows from the given layer. This also requires us to remove the
		// corresponding columns from (layer+1)!
		removedRows = new HashSet<Integer>();
		while(removedRows.size() < weights[dropoutLayer].rows/2)
			removedRows.add((int)(Math.random() * weights[dropoutLayer].rows));
		
		doDropout();
	}
	
	private void doDropout() {
		if(options.logDropout)
			options.log.info("Dropout from layer " + (dropoutLayer+1) + " neurons: " +  removedRows);
		
		dropoutWeight = weights[dropoutLayer];
		dropoutBias = biases[dropoutLayer];
		dropoutWeightPlusOne = weights[dropoutLayer+1];
		
		weights[dropoutLayer] = dropoutWeight.withoutRows(removedRows);
		biases[dropoutLayer] = dropoutBias.withoutRows(removedRows);
		weights[dropoutLayer+1] = dropoutWeightPlusOne.withoutColumns(removedRows);
	}
	
	/**
	 * Switch dropouts: restore the neurons that were dropped out, and drop out the ones that weren't.
	 */
	public void invertDropout() {
		Set<Integer> rowsNotRemoved = new HashSet<Integer>();
		for(int i=0; i<weights[dropoutLayer].rows*2; i++) {
			if(!removedRows.contains(i))
				rowsNotRemoved.add(i);
		}
		
		restore();
		
		removedRows = rowsNotRemoved;
		doDropout();
	}
	
	/**
	 * Restore dropped-out neurons.
	 */
	public void restore() {
		dropoutWeight.restoreRows(weights[dropoutLayer], removedRows);
		dropoutBias.restoreRows(biases[dropoutLayer], removedRows);
		dropoutWeightPlusOne.restoreColumns(weights[dropoutLayer+1], removedRows);
		
		weights[dropoutLayer] = dropoutWeight;
		biases[dropoutLayer] = dropoutBias;
		weights[dropoutLayer+1] = dropoutWeightPlusOne;
		
		removedRows = null;
	}
}
