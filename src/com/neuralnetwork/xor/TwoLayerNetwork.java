package com.neuralnetwork.xor;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;

public class TwoLayerNetwork
{
    /**
     * The layers indexed by training example and layer
     */
    protected LayorInfo[][] aExampleLayers;
    protected NVector[] aInput;
    protected NVector[] aExpected;

    protected NVector vError;

    final protected double alpha; /** momentum **/
    final protected double eta; /** learning parameter **/

    protected IActivationFunction.IDifferentiableFunction phi;
    protected SingleLayorNeuralNetwork firstLayer;
    protected SingleLayorNeuralNetwork secondLayer;

    public TwoLayerNetwork(Builder builder)
    {
        this.eta = builder.eta;
        this.alpha = builder.alpha;
        this.phi = builder.phi;
        this.firstLayer = builder.firstPass;
        this.secondLayer = builder.secondPass;
    }

    static public class Builder
    {
        private IActivationFunction.IDifferentiableFunction phi;
        private SingleLayorNeuralNetwork firstPass;
        private SingleLayorNeuralNetwork secondPass;
        protected Double alpha;
        protected Double eta;

        public Builder setGlobalActivationFunction(IActivationFunction.IDifferentiableFunction phi)
        {
            this.phi = phi;
            return this;
        }

        public Builder setFirstLayer(SingleLayorNeuralNetwork firstLayer)
        {
            this.firstPass = firstLayer;
            return this;
        }

        public Builder setSecondLayer(SingleLayorNeuralNetwork secondLayer)
        {
            this.secondPass = secondLayer;
            return this;
        }

        public Builder setLearningParam(Double alpha)
        {
            this.alpha = alpha;
            return this;
        }

        public Builder setMomentumParam(Double eta)
        {
            this.eta = eta;
            return this;
        }
    }

    protected void initializeExampleLayer(int layer)
    {
        aExampleLayers[layer] = new LayorInfo[secondLayer != null ? 2 : 1];

        aExampleLayers[layer][0] = new LayorInfo(firstLayer);
        if (secondLayer != null) aExampleLayers[layer][1] = new LayorInfo(secondLayer);
    }

    /**
     * This class wraps the SingleLayorNeuralNetwork
     * with additional information.
     *
     * Basically it works per layer
     * and you get back info about its individual neurons
     */
    protected class LayorInfo
    {
        final public SingleLayorNeuralNetwork layer;
        /**
         * For each neuron k in the layer,
         * store v_k (induced local field).
         */
        public NVector vInducedLocalField;
        /**
         * For each neuron k in the layer,
         * store y_k (value of impulse function).
         */
        public NVector vImpulseFunction;
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

        /**
         * For each neuron k in the layer,
         * store its gradient
         */
        public NVector vGradients;

        public LayorInfo(SingleLayorNeuralNetwork layer)
        {
            this.layer = layer;
            this.vInducedLocalField = new NVector(this.layer.getNumberOfNeurons());
            this.vImpulseFunction = new NVector(this.layer.getNumberOfNeurons());
            this.aPrevWeights = new NVector[this.layer.getNumberOfNeurons()];
            this.vGradients = new NVector(this.layer.getNumberOfNeurons());
        }
    }

    protected class Output
    {
        String weights()
        {
            String rslt = "";
            for(int neuron=0; neuron< firstLayer.getNumberOfNeurons(); neuron++)
            {
                rslt += String.format("%20s | %20s %n", firstLayer.aNeurons[neuron], getNeuron(secondLayer, neuron));
            }
            return rslt;
        }

        private String getNeuron(SingleLayorNeuralNetwork network, int neuron)
        {
            return neuron < network.getNumberOfNeurons()
                    ? network.aNeurons[neuron].toString()
                    : "";
        }
    }

    public void backpropagation(double errorBound, NVector... aInputExpected)
    {
        if (aInputExpected.length % 2 != 0)
            throw new IllegalArgumentException();

        //initialization
        initLayers(aInputExpected);

        int len = 1;

        while( backpropagation() > errorBound && len <= 1000)
        {
            System.out.println("====================================================================");
            System.out.println("Iteration = "+len++);
            System.out.format("Weights =%n%s", new Output().weights());
            System.out.format("Error = %s %n", vError.sumOfCoords());
            System.out.format("%5s %20s %20s%n", "i", "Expected", "Actual");
            for(int i=0; i<aExampleLayers.length; ++i)
                System.out.format("%5s %20s %20s%n",
                                  i,
                                  aExpected[i],
                                  aExampleLayers[i][aExampleLayers[i].length-1].vImpulseFunction);
        }
    }

    public NVector output(NVector input)
    {
        aExampleLayers = new LayorInfo[1][];
        aInput = new NVector[aExampleLayers.length];
        aExpected = new NVector[aExampleLayers.length];
        vError = new NVector(aExampleLayers.length);

        for(int i=0; i<aExampleLayers.length; i++)
        {
            initializeExampleLayer(i);
            aInput[i] = input;
        }

        return output(0,0,input);
    }

    /**
     * Trains the network using pairs of inputs/expected values
     *
     * @return error
     */
    protected double backpropagation()
    {
        for(int i=0; i<aExampleLayers.length; i++)
        {
            constructErrorFunction(i);
        }

        for(int i=0; i<aExampleLayers.length; i++)
        {
            constructGradients(i);
        }

        for(int i=0; i<aExampleLayers.length; i++)
        {
            adjustWeights(i);
        }

        return vError.sumOfCoords();
    }

    protected void initLayers(NVector... aInputExpected)
    {
        aExampleLayers = new LayorInfo[aInputExpected.length / 2][];
        aInput = new NVector[aExampleLayers.length];
        aExpected = new NVector[aExampleLayers.length];
        vError = new NVector(aExampleLayers.length);

        //save aInputExpected
        for(int i=0; i<aExampleLayers.length; i++)
        {
            initializeExampleLayer(i);
            aInput[i] = aInputExpected[2*i];
            aExpected[i] = aInputExpected[2*i+1];
        }
    }

    protected void constructErrorFunction(int example)
    {
        NVector actual = output(example, 0, aInput[example]);
        vError.set(example, aExpected[example].subtract(actual).error());
    }

    /**
     * Find the actual output for the given example.
     *
     * Side effect: store stuff (induced local field, impulse function, etc) in the layer info
     *
     * @param example index to example in aExampleLayers
     * @param layer the example's layer
     * @return output
     */
    protected NVector output(int example, int layer, NVector input)
    {
        LayorInfo layorInfo = aExampleLayers[example][layer];

        //save info for back propagation
        layorInfo.vInput = new NVector(input, 1f); //tack on bias at end
        constructInducedLocalField(layorInfo);
        constructImpulseFunction(layorInfo);

        if (layer < aExampleLayers[example].length - 1)
        {
            return output(example, layer+1, aExampleLayers[example][layer].vImpulseFunction);
        }
        return layorInfo.vImpulseFunction;
    }

    private void constructInducedLocalField(LayorInfo layorInfo)
    {
        //calculate v_k's
        int len=0;
        for(Neuron neuron:layorInfo.layer)
        {
            layorInfo.vInducedLocalField.set(len++, neuron.rawoutput(layorInfo.vInput));
        }
    }

    private void constructImpulseFunction(LayorInfo layorInfo)
    {
        //calculate y_k's aka output
        int len=0;
        for(Neuron neuron:layorInfo.layer)
        {
            layorInfo.vImpulseFunction.set(len, neuron.phi().apply(layorInfo.vInducedLocalField.get(len++)));
        }
    }

    protected void constructGradients(int example)
    {
       constructGradients(example, aExampleLayers[example].length - 1);
    }

    protected void constructGradients(int example, int layer)
    {
        if (layer >= 0)
        {
            LayorInfo layorInfo = aExampleLayers[example][layer];

            int neuronPos = 0;
            for(Neuron neuron:layorInfo.layer)
            {
                layorInfo.vGradients.set(neuronPos, gradient(example, layer, neuronPos));

                neuronPos++;
            }

            constructGradients(example, layer - 1);
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
    protected double gradient(int example, int layerLevel, int neuron)
    {
        if (layerLevel == aExampleLayers[example].length-1)
        {
            // (oj - tj) * phi'_j(v^L_j)
            final double inducedLocalField = aExampleLayers[example][layerLevel].vInducedLocalField.get(neuron);
            final double impulseFunction = aExampleLayers[example][layerLevel].vImpulseFunction.get(neuron);
            return (aExpected[example].get(neuron) - impulseFunction)
                    * phi.derivative(inducedLocalField);
        }
        else
        {
             return phi.derivative( aExampleLayers[example][layerLevel].vInducedLocalField.get(neuron))
                     * sumGradients(example, layerLevel + 1, neuron);
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
    protected double sumGradients(int example, int layerLevel, int currentNeuron)
    {
        LayorInfo layorInfo = aExampleLayers[example][layerLevel];
        double rslt = 0f;

        int neuronPos = 0;
        for(Neuron neuron:layorInfo.layer)
        {
            rslt += neuron.getWeight(currentNeuron) * gradient(example, layerLevel, neuronPos++);
        }
        return rslt;
    }

    protected void adjustWeights(int example)
    {
        adjustWeights(example, aExampleLayers[example].length - 1);
    }

    protected void adjustWeights(int example, int layer)
    {
        if (layer >= 0)
        {
            LayorInfo layorInfo = aExampleLayers[example][layer];

            int neuronPos = 0;
            for(Neuron neuron:layorInfo.layer)
            {
                if (layorInfo.aPrevWeights[neuronPos] == null)
                    layorInfo.aPrevWeights[neuronPos] = new NVector(neuron.getNumberOfWeights());

                //iterate thru the neuron's weights
                for(int weight=0; weight<neuron.getNumberOfWeights(); weight++)
                {
                    double newWeight = neuron.getWeight(weight)
                            + alpha * layorInfo.aPrevWeights[neuronPos].get(weight) //momentum
                            + eta * layorInfo.vGradients.get(neuronPos) * layorInfo.vInput.get(weight); //delta correction

                    //backup previous weight
                    layorInfo.aPrevWeights[neuronPos].set(weight, neuron.getWeight(weight));

                    neuron.setWeight(weight, newWeight);
                }

                neuronPos++;
            }

            adjustWeights(example, layer - 1);
        }
    }


    public NVector rawoutput(NVector input)
    {
        throw new NotImplementedException();
    }

//    @Override
//    public int getNumberOfNeurons()
//    {
//        return 3;
//    }


}
