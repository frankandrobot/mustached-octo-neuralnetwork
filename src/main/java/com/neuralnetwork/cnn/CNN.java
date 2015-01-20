package com.neuralnetwork.cnn;

import com.neuralnetwork.core.interfaces.INeuralNetwork;
import org.ejml.data.DenseMatrix64F;

public class CNN implements INeuralNetwork<DenseMatrix64F[],DenseMatrix64F>
{
    protected CNNLayer[] aLayers;


    CNN(CNNBuilder builder)
    {
        aLayers = builder.layers.toArray(new CNNLayer[builder.layers.size()]);
    }


    @Override
    public DenseMatrix64F generateOutput(DenseMatrix64F... inputs)
    {
        for(MapNode inputMap:aLayers[0].lMaps)
        {
            for(DenseMatrix64F input:inputs)
            {
                MapNode _inputMap = new MapNode(input);
                inputMap.addInput(_inputMap);
            }
        }


        for(CNNLayer curLayer:aLayers)
        {
            for(MapNode mapNode :curLayer.lMaps)
            {
                mapNode.generateOutput();
            }
        }

        return aLayers[aLayers.length-1].lMaps.get(0).yOutput;
    }


    @Override
    public DenseMatrix64F generateYoutput(DenseMatrix64F... denseMatrix64F)
    {
        return null;
    }
}
