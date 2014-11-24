package kricket.neural.util;

import kricket.neural.cnn.Layer;

public class IncompatibleLayerException extends Exception {
	public IncompatibleLayerException(Dimension inputDimension, Layer layer) {
		super(layer + " is incompatible with input Dimension " + inputDimension);
	}
	
	public IncompatibleLayerException(Layer lastLayer, Dimension outputDimension) {
		super("The last layer " + lastLayer + " will not produce output of Dimension " + outputDimension);
	}

	private static final long serialVersionUID = 1L;
}
