package com.neuralnetwork.core;

import com.neuralnetwork.core.interfaces.INeuralNetwork;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.util.Iterator;

public class SingleLayerNeuralNetwork implements INeuralNetwork<NVector,NVector,Neuron>, Iterable<Neuron>
{
    protected Neuron[] aNeurons;
    /**
     * Used for scratch
     */
    private NVector vLatestOutput;

    public SingleLayerNeuralNetwork() { }

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
    public NVector constructInducedLocalField(NVector input)
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
    public NVector constructOutput(NVector input)
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

    public Neuron getNeuron(int neuron) { return aNeurons[neuron]; }

}
