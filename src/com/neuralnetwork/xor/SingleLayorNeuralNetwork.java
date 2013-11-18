package com.neuralnetwork.xor;

public class SingleLayorNeuralNetwork implements INeuralNetwork
{
    Neuron[] aNeurons;
    NVector vLatestOutput;

    public SingleLayorNeuralNetwork(int size)
    {
        aNeurons = new Neuron[size];
        vLatestOutput = new NVector(size);
    }

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
    }
}
