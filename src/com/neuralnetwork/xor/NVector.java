package com.neuralnetwork.xor;

import java.util.Arrays;

public class NVector {
    float[] aCoords;

    public NVector(int size)
    {
        this.aCoords = new float[size];
    }

    public NVector(float... aCoords)
    {
        this.aCoords = Arrays.copyOf(aCoords, aCoords.length);
    }

    public NVector(NVector vector)
    {
        this(vector.aCoords);
    }

    public NVector(NVector vector, float... aCoords)
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
    public float dot(NVector input)
    {
        float rslt=0;
        for(int i=0; i<input.size(); i++)
            rslt += aCoords[i] * input.aCoords[i];
        return rslt;
    }

    public int size() { return aCoords.length; }

    public float first()
    {
        return aCoords[0];
    }

    public float last()
    {
        return aCoords[aCoords.length-1];
    }

    public NVector set(int i, float output)
    {
        this.aCoords[i] = output;
        return this;
    }

    public float get(int i)
    {
        return this.aCoords[i];
    }

    @Override
    public String toString()
    {
        StringBuilder string = new StringBuilder(100);
        string.append("[");
        for(float coord:aCoords)
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
}
