package com.neuralnetwork.xor;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.util.Iterator;

public class SingleLayorNeuralNetwork implements INeuralNetwork, Iterable<Neuron>
{
    protected Neuron[] aNeurons;
    /**
     * Used for scratch
     */
    private NVector vLatestOutput;

    public SingleLayorNeuralNetwork() { }

    /**
     * For each neuron k, let W_k be its vector of weights.
     * Then this method computes
     *    v_k = W_k . input
     * where . is the dot product
     * and saves the result in #vLatestOutput
     *
     * @param input the input
     * @return
     */
    @Override
    public NVector inducedLocalField(NVector input)
    {
        int len=0;

        for(Neuron neuron:aNeurons)
        {
            vLatestOutput.set(len++, neuron.rawoutput(input));
        }

        return vLatestOutput;
    }

    /**
     * For each neuron k, let W_k be its vector of weights and
     * let phi_k be its activation function.
     * Then this method computes
     *     y_k = phi_k( W_k . input)
     * where . is the dot product
     * and saves the result in #vLatestOutput
     *
     * @param input
     * @return
     */
    public NVector output(NVector input)
    {
        int len=0;

        for(Neuron neuron:aNeurons)
        {
            vLatestOutput.set(len++, neuron.output(input));
        }

        return vLatestOutput;
    }

    public int getNumberOfNeurons() { return this.aNeurons.length; }

    public void setNeurons(Neuron... aNeurons)
    {
        this.aNeurons = aNeurons;
        this.vLatestOutput = new NVector().setSize(aNeurons.length);
    }

    @Override
    public Iterator<Neuron> iterator()
    {
        return new Iterator<Neuron>() {

            int len=0;

            @Override
            public boolean hasNext() {
                return len < aNeurons.length;
            }

            @Override
            public Neuron next() {
                return aNeurons[len++];
            }

            @Override
            public void remove()
            {
                throw new NotImplementedException();
            }
        };
    }

    public NVector weights()
    {
        NVector rslt = new NVector(aNeurons[0].getWeights());
        for(Neuron neuron:aNeurons)
        {
            if (neuron != aNeurons[0])
            {
                rslt = rslt.concatenate(neuron.getWeights());
            }
        }

        return rslt;
    }

}
