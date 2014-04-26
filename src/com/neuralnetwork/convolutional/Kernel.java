package com.neuralnetwork.convolutional;

import org.hamcrest.Matcher;
import org.junit.Assert;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import static org.hamcrest.CoreMatchers.is;

public class Kernel
{
    private int height;
    private int width;
    private double[] matrix;

    /**
     * Accepts a matrix in column-major format
     *
     * @param cols
     * @param rows
     * @param matrix
     */
    public Kernel(int cols, int rows, double[] matrix)
    {
        Assert.assertThat(matrix.length, is(cols * rows));

        this.matrix = matrix;
        this.height = rows;
        this.width = cols;

    }

    public double[] getKernelData()
    {
        return matrix;
    }

    public int getHeight() {
        return height;
    }

    public int getWidth() {
        return width;
    }
}
