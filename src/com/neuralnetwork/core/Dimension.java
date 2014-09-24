package com.neuralnetwork.core;

public class Dimension
{
    final public int rows, cols;

    public Dimension(int r, int c)
    {
        this.rows = r;
        this.cols = c;
    }

    @Override
    public boolean equals(Object o)
    {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        Dimension dimension = (Dimension) o;

        if (cols != dimension.cols) return false;
        if (rows != dimension.rows) return false;

        return true;
    }
}
