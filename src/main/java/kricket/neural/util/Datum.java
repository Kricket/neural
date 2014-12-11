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
}
