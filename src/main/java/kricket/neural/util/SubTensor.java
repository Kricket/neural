package kricket.neural.util;

public class SubTensor {
	public final Tensor source;
	public final int rowOff, colOff, sliceOff, rows, cols, slices;
	
	public SubTensor(Tensor source, int startRow, int startCol, int startSlice, int rows, int cols, int slices) {
		this.source = source;
		rowOff = startRow;
		colOff = startCol;
		sliceOff = startSlice;
		this.rows = rows;
		this.cols = cols;
		this.slices = slices;
	}
	
	public double at(int row, int col, int slice) {
		return source.at(row+rowOff, col+colOff, slice + sliceOff);
	}
	
	public void set(int row, int col, int slice, double val) {
		source.set(row + rowOff, col + colOff, slice + sliceOff, val);
	}
	
	public int index(int row, int col, int slice) {
		return source.index(row+rowOff, col+colOff, slice + sliceOff);
	}
	
	/**
	 * Compute the inner product (i.e. the sum of the products of the entries) of this
	 * with the given Tensor.
	 * @param m
	 * @param mSlice
	 * @return
	 */
	public double innerProduct(Tensor m) {
		if(m.rows != rows || m.cols != cols || slices != m.slices)
			throw new IllegalArgumentException("Incompatible dimensions");
		
		double sum = 0;
		int mOff = 0;
		for(int s=0; s<slices; s++) for(int r=0; r<rows; r++) for(int c=0; c<cols; c++) {
			sum += at(r,c,s) * m.data[mOff++];
		}
		return sum;
	}

	/**
	 * Set this = this + (t*d)
	 * @param t
	 * @param d
	 */
	public void plusEqualsTimes(Tensor t, double d) {
		if(t.rows != rows || t.cols != cols || slices != t.slices)
			throw new IllegalArgumentException("Incompatible dimensions");
		
		int mOff = 0;
		for(int s=0; s<slices; s++) for(int r=0; r<rows; r++) for(int c=0; c<cols; c++) {
			source.data[index(r, c, s)] += t.data[mOff++] * d;
		}
	}
	
	@Override
	public String toString() {
		return getClass().getSimpleName()
				+ " starting at ("
				+ rowOff + ", " + colOff + ", " + sliceOff
				+ ") size ("
				+ rows + ", " + cols + ", " + slices
				+ ")";
	}
}
