package com.neuralnetwork.xor;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;

public class TwoLayerNetwork
{
    protected LayorInfo[] aLayers;

    protected NVector[] aExampleInput;
    protected NVector[] aExpected;
    protected NVector[] aActual;
    protected NVector vError;
    protected int numberExamples;

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

        public Builder setMomentumParam(Double alpha)
        {
            this.alpha = alpha;
            return this;
        }

        public Builder setLearningParam(Double eta)
        {
            this.eta = eta;
            return this;
        }
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
         * store value of impulse function.
         */
        public NVector vImpulseFunction;
        /**
         * Store the inputs (y_k) from the previous layer.
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
            this.vInducedLocalField = new NVector().setSize(this.layer.getNumberOfNeurons());
            this.vImpulseFunction = new NVector().setSize(this.layer.getNumberOfNeurons());
            this.aPrevWeights = new NVector[this.layer.getNumberOfNeurons()];
            this.vGradients = new NVector().setSize(this.layer.getNumberOfNeurons());
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

    protected void initializeLayers()
    {
        aLayers = new LayorInfo[secondLayer != null ? 2 : 1];

        aLayers[0] = new LayorInfo(firstLayer);
        if (secondLayer != null) aLayers[1] = new LayorInfo(secondLayer);
    }

    protected void initLayers(NVector... aInputExpected)
    {
        if (aInputExpected.length % 2 != 0)
            throw new IllegalArgumentException();

        initializeLayers();

        numberExamples = aInputExpected.length / 2;

        aExampleInput = new NVector[numberExamples];
        aExpected = new NVector[numberExamples];
        vError = new NVector().setSize(numberExamples);
        aActual = new NVector[numberExamples];

        //save aInputExpected
        for(int i=0; i< numberExamples; i++)
        {
            aExampleInput[i] = aInputExpected[2*i];
            aExpected[i] = aInputExpected[2*i+1];
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
            len = debugOutput(len);
        }
        debugOutput(len);
    }

    protected int debugOutput(int len)
    {
        System.out.println("====================================================================");
        System.out.println("Iteration = "+len++);
        System.out.format("Weights =%n%s", new Output().weights());
        System.out.format("Error = %s %n", vError.sumOfCoords());
        System.out.format("%5s %20s %20s%n", "i", "Expected", "Actual");
        for(int i=0; i< numberExamples; ++i)
            System.out.format("%5s %20s %20s%n",
                              i,
                              aExpected[i],
                              aActual[i]);
        return len;
    }

    public NVector output(NVector input)
    {
        if (aLayers == null)
        {
            initializeLayers();
            numberExamples = 1;

            aExampleInput = new NVector[numberExamples];
            aExpected = new NVector[numberExamples];
            vError = new NVector().setSize(numberExamples);
            aActual = new NVector[numberExamples];
        }

        aExampleInput[0] = input;

        return output(0, 0, input);
    }

    /**
     * Trains the network using pairs of inputs/expected values
     *
     * @return dotProduct
     */
    protected double backpropagation()
    {
        for(int i=0; i< numberExamples; i++)
        {
            forwardPropagation(i);
            constructGradients(i);
            adjustWeights();
        }

        for(int i=0; i< numberExamples; i++)
        {
            constructErrorFunction(i);
        }


        return vError.sumOfCoords();
    }

    protected void constructErrorFunction(int example)
    {
        NVector actual = forwardPropagation(example);
        vError.set(example, aExpected[example].subtract(actual).dotProduct());
    }

    protected NVector forwardPropagation(int example)
    {
        return output(example, 0, aExampleInput[example]);
    }

    /**
     * Find the actual output for the given example.
     *
     * Side effect: store stuff (induced local field, impulse function, etc) in the layer info
     *
     *
     * @param example
     * @param layer the example's layer
     * @return output
     */
    protected NVector output(int example, int layer, NVector input)
    {
        LayorInfo layorInfo = aLayers[layer];

        //save info for back propagation
        layorInfo.vInput = new NVector(input, 1f); //tack on bias at end

        constructInducedLocalField(layorInfo);
        constructImpulseFunction(layorInfo);

        if (layer < aLayers.length - 1)
        {
            return output(example, layer+1, aLayers[layer].vImpulseFunction);
        }
        else
        {
            aActual[example] = new NVector(aLayers[layer].vImpulseFunction);
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
       constructGradients(example, aLayers.length - 1);
    }

    protected void constructGradients(int example, int layer)
    {
        if (layer >= 0)
        {
            LayorInfo layorInfo = aLayers[layer];

            for(int neuronPos=0; neuronPos<layorInfo.layer.getNumberOfNeurons(); neuronPos++)
            {
                layorInfo.vGradients.set(neuronPos, gradient(example, layer, neuronPos));
            }

            constructGradients(example, layer - 1);
        }
    }

    /**
     * Gradient for the given example, layer, and neuron
     *
     * @param example example
     * @param layerLevel layer level
     * @param neuron neuron
     * @return gradient value for the given example, layer, and neuron
     */
    protected double gradient(int example, int layerLevel, int neuron)
    {
        if (layerLevel == aLayers.length-1)
        {
            // (oj - tj) * phi'_j(v^L_j)
            final double inducedLocalField = aLayers[layerLevel].vInducedLocalField.get(neuron);
            final double impulseFunction = aLayers[layerLevel].vImpulseFunction.get(neuron);
            return (aExpected[example].get(neuron) - impulseFunction)
                    * phi.derivative(inducedLocalField);
        }
        else
        {
             return phi.derivative( aLayers[layerLevel].vInducedLocalField.get(neuron))
                     * sumGradients(example, layerLevel + 1, neuron);
        }
    }

    /**
     * Find the sum of the gradients times the weights of the next layer
     * for the current neuron
     *
     *
     *
     * @param example
     * @param layerLevel layer level
     * @param currentNeuron current neuron
     * @return
     */
    protected double sumGradients(int example, int layerLevel, int currentNeuron)
    {
        LayorInfo layorInfo = aLayers[layerLevel];
        double rslt = 0f;

        int neuronPos = 0;
        for(Neuron neuron:layorInfo.layer)
        {
            rslt += neuron.getWeight(currentNeuron) * gradient(example, layerLevel, neuronPos++);
        }
        return rslt;
    }

    protected void adjustWeights()
    {
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
                    layorInfo.aPrevWeights[neuronPos] = new NVector().setSize(neuron.getNumberOfWeights());

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

            adjustWeights(layer - 1);
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
