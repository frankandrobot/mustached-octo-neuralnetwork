package com.neuralnetwork.core;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.util.Arrays;
import java.util.Iterator;

public class NVector implements Iterable<Double>
{
    double[] aCoords;

    /**
     * Make sure to call #setSize
     */
    public NVector() {}

    public NVector setSize(int i)
    {
        if (aCoords == null) aCoords = new double[i];
        else aCoords = Arrays.copyOf(aCoords, i);
        return this;
    }

    public NVector(double... aCoords)
    {
        this.aCoords = Arrays.copyOf(aCoords, aCoords.length);
    }

    public NVector(NVector vector)
    {
        this(vector.aCoords);
    }

    public NVector(NVector vector, double... aCoords)
    {
        this.aCoords = Arrays.copyOf(vector.aCoords, vector.aCoords.length + aCoords.length);
        for(int i=0; i<aCoords.length; i++)
            this.aCoords[vector.aCoords.length + i] = aCoords[i];
    }

    /**
     * @warning assumes input.size() <= this.size()!
     *
     * @param input vector
     * @return dot product
     */
    public double dot(NVector input)
    {
        double rslt=0;
        for(int i=0; i<input.size(); i++)
            rslt += aCoords[i] * input.aCoords[i];
        return rslt;
    }

    public int size() { return aCoords.length; }

    public double first()
    {
        return aCoords[0];
    }

    public double last()
    {
        return aCoords[aCoords.length-1];
    }

    public NVector set(int i, double output)
    {
        this.aCoords[i] = output;
        return this;
    }

    public double get(int i)
    {
        return this.aCoords[i];
    }

    @Override
    public String toString()
    {
        StringBuilder string = new StringBuilder(100);
        string.append("[");
        for(int i=0; i<aCoords.length; ++i)
        {
            string.append(String.format("%6.6g",aCoords[i]));
            if (i<aCoords.length-1) string.append("  ");
        }
        string.append("]");
        return string.toString();
    }

    public NVector subtract(NVector vector)
    {
        NVector rslt = new NVector(this);
        for(int i=0; i<aCoords.length; i++)
            rslt.aCoords[i] -= vector.aCoords[i];
        return rslt;
    }

    /**
     * Computes this.this
     *
     * Ex: if vector is 3D, then dotProduct() = x^2 + y^2 + z^2
     *
     * @return this.this
     */
    public double dotProduct()
    {
        double rslt = 0f;
        for(int i=0; i<aCoords.length; i++)
            rslt += aCoords[i] * aCoords[i];
        return rslt;
    }

    /**
     * Sum of coordinates
     *
     * @return double
     */
    public double sumOfCoords()
    {
        double rslt = 0f;
        for(int i=0; i<aCoords.length; i++)
            rslt += aCoords[i];
        return rslt;
    }

    public NVector concatenate(NVector weights)
    {
        return new NVector(this, weights.aCoords);
    }

    @Override
    public Iterator<Double> iterator()
    {
        return new Iterator<Double>() {
            int len = 0;

            @Override
            public boolean hasNext()
            {
                return len < aCoords.length;
            }

            @Override
            public Double next()
            {
                return aCoords[len++];
            }

            @Override
            public void remove()
            {
                throw new NotImplementedException();
            }
        };
    }
}
