package com.neuralnetwork.xor;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.util.Random;

public class XORBackPropagationNetwork implements INeuralNetwork
{
    LayorInfo[] aLayers;
    NVector vError;

    protected static float alpha = 0.5f;
    protected static float eta = 0.5f;

    IActivationFunction.IDifferentiableFunction phi;

    public XORBackPropagationNetwork()
    {
        phi = new IActivationFunction.SigmoidFunction(1f);

        SingleLayorNeuralNetwork firstPass = new SingleLayorNeuralNetwork();
        SingleLayorNeuralNetwork secondPass = new SingleLayorNeuralNetwork();

        Random r = new Random();

        firstPass.setNeurons(new Neuron(phi, r.nextFloat(), r.nextFloat(), r.nextFloat()),
                new Neuron(phi, r.nextFloat(), r.nextFloat(), r.nextFloat()));

        secondPass.setNeurons(new Neuron(phi, r.nextFloat(), r.nextFloat(), r.nextFloat()));

        aLayers = new LayorInfo[2];

        aLayers[0] = new LayorInfo(firstPass);
        aLayers[1] = new LayorInfo(secondPass);
    }

    /**
     * This class wraps the SingleLayorNeuralNetwork
     * with additional information.
     *
     * Basically it works per layer
     * and you get back its individual neurons
     * and in turn get each individual neurons weights
     */
    protected class LayorInfo
    {
        final public SingleLayorNeuralNetwork layer;
        /**
         * For each neuron k in the layer,
         * store v_k.
         */
        public NVector vRawoutput;
        /**
         * For each neuron k in the layer,
         * store y_k
         */
        public NVector vOutput;
        /**
         * Store the inputs from the previous layer.
         * vInput[i] is matched with a neuron's ith weight w_i
         * to produce the output i.e.,
         * vInput[i] = y^(l-1)_i
         */
        public NVector vInput;
        /**
         * For each neuron k in the layer,
         * aPrevWeights[k] is its previous set of weights
         */
        public NVector[] aPrevWeights;

        public LayorInfo(SingleLayorNeuralNetwork layer)
        {
            this.layer = layer;
            this.vRawoutput = new NVector(this.layer.getNumberOfNeurons());
            this.vOutput = new NVector(this.layer.getNumberOfNeurons());
            this.aPrevWeights = new NVector[this.layer.getNumberOfNeurons()];
        }
    }

    @Override
    public NVector output(NVector input)
    {
        return output(0, input);
    }

    protected NVector output(int layer, NVector input)
    {
        LayorInfo layorInfo = aLayers[layer];

        //save info for back propagation
        layorInfo.vInput = new NVector(input, 1f); //tack on bias at end
        calculateRawOutput(layorInfo); //calculate v_k's
        calculateOutput(layorInfo);

        if (layer < aLayers.length - 1)
        {
            return output(layer+1, aLayers[layer].vOutput);
        }
        return layorInfo.vOutput;
    }

    private void calculateRawOutput(LayorInfo layorInfo)
    {
        //calculate v_k's
        int len=0;
        for(Neuron neuron:layorInfo.layer)
        {
            layorInfo.vRawoutput.set(len++, neuron.rawoutput(layorInfo.vInput));
        }
    }

    private void calculateOutput(LayorInfo layorInfo)
    {
        //calculate y_k's
        int len=0;
        for(Neuron neuron:layorInfo.layer)
        {
            layorInfo.vOutput.set(len, neuron.phi().apply(layorInfo.vRawoutput.get(len++)));
        }
    }

    public void backpropagation(NVector input, NVector expected)
    {
        NVector actual = output(input);
        vError = expected.subtract(actual);

        adjustWeights(aLayers.length - 1);
    }

    protected void adjustWeights(int layer)
    {
        if (layer >= 0)
        {
            LayorInfo layorInfo = aLayers[layer];

            int neuronPos = 0;
            for(Neuron neuron:layorInfo.layer)
            {
                if (layorInfo.aPrevWeights[neuronPos] == null)
                    layorInfo.aPrevWeights[neuronPos] = new NVector(neuron.getNumberOfWeights());

                //iterate thru the neuron's weights
                for(int weight=0; weight<neuron.getNumberOfWeights(); weight++)
                {
                    float newWeight = neuron.getWeight(weight)
                                      + alpha * layorInfo.aPrevWeights[neuronPos].get(weight)
                                      + eta * gradient(layer, neuronPos) * layorInfo.vInput.get(weight);

                    //backup previous weight
                    layorInfo.aPrevWeights[neuronPos].set(weight, neuron.getWeight(weight));

                    neuron.setWeight(weight, newWeight);
                }

                neuronPos++;
            }

            adjustWeights(layer - 1);
        }
    }

    /**
     * Given a network layor and the neuron's position (in the array)
     * return its gradient.
     *
     * @param layerLevel
     * @param neuron
     * @return
     */
    private float gradient(int layerLevel, int neuron)
    {
        if (layerLevel == aLayers.length-1)
        {
            // e^L_j * phi'_j(v^L_j)
            return vError.get(neuron) * phi.derivative( aLayers[layerLevel].vRawoutput.get(neuron) );
        }
        else
        {
             return phi.derivative( aLayers[layerLevel].vRawoutput.get(neuron))
                     * sumGradients(layerLevel + 1, neuron);
        }
    }

    /**
     * Find the sum of the gradients times the weights of the next layer
     * for the current neuron
     *
     * @param layerLevel layer level
     * @param currentNeuron current neuron
     * @return
     */
    private float sumGradients(int layerLevel, int currentNeuron)
    {
        LayorInfo layorInfo = aLayers[layerLevel];
        float rslt = 0f;

        int neuronPos = 0;
        for(Neuron neuron:layorInfo.layer)
        {
            rslt += neuron.getWeight(currentNeuron) * gradient(layerLevel, neuronPos++);
        }
        return rslt;
    }

    public NVector rawoutput(NVector input)
    {
        throw new NotImplementedException();
    }

    @Override
    public int getNumberOfNeurons()
    {
        return 3;
    }


}
