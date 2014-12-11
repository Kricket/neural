package kricket.neural.util;

/**
 * I'm lying - this is actually a (sub-)matrix!
 */
public class SubTensor {
	private final Tensor source;
	private final int sliceOff, rowOff, colOff, rows, cols;
	
	public SubTensor(Tensor source, int sliceOff, int startRow, int startCol, int rows, int cols) {
		this.source = source;
		this.sliceOff = sliceOff;
		rowOff = startRow;
		colOff = startCol;
		this.rows = rows;
		this.cols = cols;
	}
	
	public double at(int row, int col) {
		return source.at(row+rowOff, col+colOff, sliceOff);
	}
	
	public void set(int row, int col, double val) {
		source.set(row + rowOff, col + colOff, sliceOff, val);
	}
	
	public int index(int row, int col) {
		return source.index(row+rowOff, col+colOff, sliceOff);
	}
	
	/**
	 * Compute the inner product (i.e. the sum of the products of the entries) of this
	 * with a slice of the given Tensor.
	 * @param m
	 * @param mSlice
	 * @return
	 */
	public double innerProduct(Tensor m, int mSlice) {
		if(m.rows != rows || m.cols != cols)
			throw new IllegalArgumentException("Incompatible dimensions");
		
		double sum = 0;
		int mOff = m.index(0, 0, mSlice);
		for(int r=0; r<rows; r++) for(int c=0; c<cols; c++) {
			sum += at(r,c) * m.data[mOff++];
		}
		return sum;
	}

	/**
	 * Set this = this + (t[slice] * d)
	 * @param t
	 * @param slice
	 * @param d
	 */
	public void plusEqualsSliceTimes(Tensor t, int slice, double d) {
		if(rows != t.rows || cols != t.cols)
			throw new IllegalArgumentException("Incompatible dimensions");
		
		int tOff = t.index(0, 0, slice);
		for(int r=0; r<rows; r++) for(int c=0; c<cols; c++) {
			source.data[index(r,c)] += (t.data[tOff++] * d);
		}
	}

	/**
	 * Set this = this + (t*d)
	 * @param t
	 * @param d
	 */
	public void plusEqualsTimes(SubTensor t, double d) {
		if(rows != t.rows || cols != t.cols)
			throw new IllegalArgumentException("Incompatible dimensions");
		
		for(int r=0; r<rows; r++) for(int c=0; c<cols; c++) {
			source.data[index(r,c)] += (t.at(r,c) * d);
		}
	}
}
