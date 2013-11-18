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
}
