package com.neuralnetwork.xor;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.util.Iterator;

public class SingleLayorNeuralNetwork implements INeuralNetwork, Iterable<Neuron>
{
    Neuron[] aNeurons;
    NVector vLatestOutput;

    public SingleLayorNeuralNetwork() { }

    public NVector output(NVector input)
    {
        int len=0;

        for(Neuron neuron:aNeurons)
        {
            vLatestOutput.set(len++, neuron.output(input));
        }

        return vLatestOutput;
    }

    public void setNeurons(Neuron... aNeurons)
    {
        this.aNeurons = aNeurons;
        this.vLatestOutput = new NVector(aNeurons.length);
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
}
