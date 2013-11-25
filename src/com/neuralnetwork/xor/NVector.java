package com.neuralnetwork.xor;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.util.Arrays;
import java.util.Iterator;

public class NVector implements Iterable<Double>
{
    double[] aCoords;

    public NVector(int size)
    {
        this.aCoords = new double[size];
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
        for(double coord:aCoords)
        {
            string.append(coord);
            string.append(", ");
        }
        string.append("]");
        return string.toString();
    }

    public NVector set(NVector vector)
    {
        for(int i=0; i<aCoords.length; i++)
            aCoords[i] = vector.aCoords[i];
        return this;
    }

    public NVector subtract(NVector vector)
    {
        NVector rslt = new NVector(this);
        for(int i=0; i<aCoords.length; i++)
            rslt.aCoords[i] -= vector.aCoords[i];
        return rslt;
    }

    /**
     * Computes ||this.this||^2
     *
     * @return
     */
    public double error()
    {
        double rslt = 0f;
        for(int i=0; i<aCoords.length; i++)
            rslt += aCoords[i] * aCoords[i];
        return rslt;
    }

    public double mylen()
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
