package com.neuralnetwork.cnn;

import com.neuralnetwork.core.interfaces.INeuralNetwork;
import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;

import java.util.Map;

public class CNN implements INeuralNetwork<DenseMatrix64F>
{
    protected CNNLayer[] aLayers;


    CNN(CNNBuilder builder)
    {
        aLayers = builder.layers.toArray(new CNNLayer[builder.layers.size()]);
    }


    @Override
    public DenseMatrix64F generateOutput(DenseMatrix64F input)
    {
        for(MapInfo inputMap:aLayers[0].lMaps)
        {
            MapInfo _input = new MapInfo(input);
            inputMap.lInputMaps.add(_input);
        }


        for(CNNLayer curLayer:aLayers)
        {
            for(MapInfo mapInfo:curLayer.lMaps)
            {
                mapInfo.generateOutput();
            }
        }

        return aLayers[aLayers.length-1].lMaps.get(0).yOutput;
    }


    @Override
    public DenseMatrix64F generateYoutput(DenseMatrix64F denseMatrix64F)
    {
        return null;
    }
}
