package com.neuralnetwork.xor;

public class Neuron
{
    NVector vWeights;
    IActivationFunction phi;

    /**
     * For back propagation
     */
    NVector vPrevWeights, vPrevInput, vPrevRawWeights;

    public Neuron(IActivationFunction phi, NVector vWeights)
    {
        this.vWeights = vWeights;
        this.phi = phi;
    }

    public Neuron(IActivationFunction phi, float... aWeights)
    {
        this.vWeights = new NVector(aWeights);
        this.phi = phi;
    }

    public float rawoutput(NVector input)
    {
        vPrevInput = new NVector(input);
        vPrevWeights = new NVector(vWeights);

        return vWeights.dot(input)
                + vWeights.last(); //don't forget the bias
    }

    public float output(NVector input)
    {
        return phi.apply(rawoutput(input));
    }

    public float getCurWeight(int weight)
    {
        return vWeights.get(weight);
    }

    public float getPrevWeight(int weight)
    {
        return vPrevWeights.get(weight);
    }

    public float getPrevInput(int weight)
    {
        return vPrevInput.get(weight);
    }

    public int size()
    {
        return vWeights.size();
    }
}
