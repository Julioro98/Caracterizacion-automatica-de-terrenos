
// CREACION DE LA REGION GEOGRAFICA DE REFERENCIA 

var region = 
    /* color: #d63000 */
    /* displayProperties: [
      {
        "type": "rectangle"
      }
    ] */
    ee.Geometry.Polygon(
        [[[-92.16777343750002, 35.839812700888025],
          [-92.16777343750002, 33.664649121127994],
          [-90.21220703125002, 33.664649121127994],
          [-90.21220703125002, 35.839812700888025]]], null, false);

//CREACION DE LAS AREAS RECTANGULARES PARA LA EXPORTACION DE IMAGENES 
var batch = require('users/fitoprincipe/geetools:batch')

//Cantidad de puntos dentro de la region seleccionada. 
var randompoints = ee.FeatureCollection.randomPoints(region, 50);

//Aumento o disminucion del punto aleatorio 
var bufferPoly = function(feature) {
  return feature.buffer(3000);          //Tamaño del area del poligono. 
};

var buffers = randompoints.map(bufferPoly); //A los puntos se les aplica el cambio de tamaño.

//Funcion para poner el cuadrado encima del circulo 
var bounding_box_func = function(feature) {
    var intermediate_buffer = feature.buffer(1500);  // Area del cuadrado 
    var intermediate_box = intermediate_buffer.bounds(); // Caja alrededor del circulo
       return(intermediate_box); // Retornar la caja 
      };

//Se utilizan la funcion bounding para poner los cuadrados a los buffers. 
var bounding_boxes = buffers.map(bounding_box_func);
var cajas = ee.FeatureCollection(bounding_boxes);

print(cajas);
Map.addLayer(cajas, {}, "cajas");

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//BASE DATOS DE CROPLAND
var dataset = ee.ImageCollection("USDA/NASS/CDL")
                  .filter(ee.Filter.date('2018-01-01', '2018-12-31'))
                  .first();
var cropLandcover = dataset.select('cropland');

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//BASE DE DATOS LANDSAT 
var dataset = ee.ImageCollection('USDA/NAIP/DOQQ')
                  .filter(ee.Filter.date('2018-01-01', '2018-12-31'));

var high = dataset.median(); 

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

//CAPTURA INDIVIDUAL DE CADA AREA EN UNA COLECCION DE IMAGENES  
var Cropland = ee.ImageCollection(cajas.map(
  function(c)
  {
  return cropLandcover.clip(c);
  }
));

var HighRes = ee.ImageCollection(cajas.map(
  function(c)
  {
  return high.clip(c);
  }
));

Map.addLayer(Cropland, {}, 'Cropland');
print(Cropland,'Cropland');

Map.addLayer(HighRes, {}, 'HighRes');
print(HighRes,'HighRes');

// EXPORTACION DE CADA IMAGEN HACIA GOOGLE DRIVE

batch.Download.ImageCollection.toDrive(HighRes, 'Train', 
                {scale: 30, 
                 region: null, 
                 type: 'double'})

batch.Download.ImageCollection.toDrive(Cropland, 'Train', 
                {scale: 30, 
                 region: null, 
                 type: 'uint8'})
