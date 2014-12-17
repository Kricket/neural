package kricket.neural.cnn;

import java.util.ArrayList;
import java.util.List;

import kricket.neural.NNBase;
import kricket.neural.nn.NN;
import kricket.neural.util.Datum;
import kricket.neural.util.Dimension;
import kricket.neural.util.IncompatibleLayerException;
import kricket.neural.util.NNOptions;
import kricket.neural.util.Tensor;

/**
 * A rewrite of the {@link NN} class. This class takes a more object-oriented approach,
 * allowing us to specify each layer independently.
 */
public class CNN extends NNBase {

	/**
	 * The layers of this CNN (not counting the input layer).
	 */
	private final Layer[] layers;
	
	/**
	 * Create a new network.
	 * <p><b>Attention: </b>an EXTRA SigmaLayer will be added to the end!
	 * @param opts
	 * @param layers
	 * @throws IncompatibleLayerException if the given layer configuration is inconsistent 
	 */
	public CNN(NNOptions opts, Dimension inputDimension, Layer ...layers) throws IncompatibleLayerException {
		super(opts);
		
		this.layers = new Layer[layers.length+1];
		for(int i=0; i<layers.length; i++)
			this.layers[i] = layers[i];
		this.layers[layers.length] = new SigmaLayer();
		
		prepare(inputDimension);
	}
	
	/**
	 * The layers of this network.
	 * @return
	 */
	public Layer[] getLayers() {
		return layers;
	}
	
	/**
	 * Verify the consistency of the Layers, and perform any pre-run optimizations.
	 * @param inputDimension
	 * @throws IncompatibleLayerException
	 */
	private void prepare(Dimension inputDimension) throws IncompatibleLayerException {
		if(options.logDimensions)
			options.log.info("Input: " + inputDimension);
		for(Layer layer : layers) {
			inputDimension = layer.prepare(inputDimension);
			if(options.logDimensions)
				options.log.info(layer.getClass().getSimpleName() + " => " + inputDimension);
		}
	}
	
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		for(Layer l:layers) {
			sb.append(l);
			sb.append("\n");
		}
		return sb.toString();
	}
	
	/**
	 * Get the result of running the given feature maps through this CNN.
	 * @param x The initial feature maps.
	 * @return
	 */
	public Tensor feedForward(Tensor x) {
		for(Layer layer : layers) {
			x = layer.feedForward(x);
		}
		return x;
	}
	
	@Override
	protected void runBatch(List<? extends Datum> batch, double regTerm, double eta) {
		for(Layer layer : layers)
			layer.resetGradients();
		
		for(Datum dat : batch)
			backprop(dat.getDataTensor(), dat.getAnswerTensor());
		
		for(Layer layer : layers)
			layer.applyGradients(regTerm, eta/batch.size());
	}

	/**
	 * Run the backpropagation algorithm.
	 * @param x
	 * @param y
	 */
	private void backprop(Tensor x, Tensor y) {
		Tensor forward = feedForward(x);
		Tensor deltas = forward.minus(y);
		
		// The cross-entropy cost function basically boils down to not running
		// backpropagation on the last (sigmoid) layer. If we instead wanted
		// to use quadratic cost, we would simply include the last layer in the
		// following loop.
		for(int i=layers.length-2; i>=0; i--) {
			deltas = layers[i].backprop(deltas);
		}
	}

	@Override
	public double calc_error(List<? extends Datum> data) {
		double numCorrect = 0;
		for(Datum dat : data) {
			Tensor result = feedForward(dat.getDataTensor());
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
	
	
	/**
	 * Experiment: attempt to pre-train the network. The idea here is:
	 * <p>The output of a fully-connected layer is a vector in N-dimensional space.
	 * What we want to do, is try to make it so that inputs of the same class are
	 * close to each other, but inputs of different classes are far apart. To do this,
	 * we simply "push" the output vectors together or apart, using the already-existing
	 * backpropagation algorithm.
	 * @param data
	 * @param batchSize
	 * @param regTerm
	 * @param eta
	 */
	public void preTrain(List<? extends Datum> data, int batchSize, double regTerm, double eta) {
		// Go up to the first fully-connected layer
		List<Layer> preLayers = new ArrayList<>();
		for(Layer l : layers) {
			preLayers.add(l);
			if(l instanceof FullyConnectedLayer)
				break;
		}
		
		eta = eta / batchSize;
		for(int start=0; start<data.size(); start+=batchSize) {
			List<? extends Datum> batch = data.subList(start, start+batchSize);
			preTrain(preLayers, batch, regTerm, eta);
		}
	}
	
	private void preTrain(List<Layer> preLayers, List<? extends Datum> batch, double regTerm, double eta) {
		for(Layer l : preLayers)
			l.resetGradients();
		
		Datum d1 = batch.get(batch.size()-1);
		Tensor y1 = forward(preLayers, d1.getDataTensor()).copy();
		for(Datum d2 : batch) {
			// First, get the vector from d1 -> d2
			Tensor y2 = forward(preLayers, d2.getDataTensor()).copy();
			Tensor diff = y2.minus(y1);
			
			if(d1.getAnswerClass() == d2.getAnswerClass())
				// Pull them together
				diff.timesEquals(-eta / diff.norm());
			else
				// Push them apart
				diff.timesEquals(eta / diff.norm());
			
			// Then, backprop to "push" d2 in the desired direction
			for(int i=preLayers.size()-1; i>=0; i--) {
				diff = preLayers.get(i).backprop(diff);
			}
			
			// Finally, setup for the next iteration.
			d1 = d2;
			y1 = y2;
		}
		
		for(Layer l : preLayers)
			l.applyGradients(regTerm, eta/batch.size());
	}

	/**
	 * Feedforward, but only using the given layers.
	 * @param preLayers
	 * @param x
	 * @return
	 */
	private Tensor forward(List<Layer> preLayers, Tensor x) {
		for(Layer l : preLayers) {
			x = l.feedForward(x);
		}
		return x;
	}
}
