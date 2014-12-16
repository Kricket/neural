package kricket.neural.util;


public interface Datum {

	/**
	 * The input data.
	 * @return
	 */
	Matrix getData();
	Tensor getDataTensor();
	/**
	 * The expected answer.
	 * @return
	 */
	Matrix getAnswer();
	Tensor getAnswerTensor();
	/**
	 * Get a short representation of the class of the answer.
	 * @return
	 */
	int getAnswerClass();
}
