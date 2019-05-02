#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include<boost/python.hpp>
#include"coder.h"
#include <numpy/arrayobject.h>
void my_encoder(Coder * coder, boost::python::object data_obj, int tncode, int tsum, int tsymbol) {
	PyArrayObject* data_arr = reinterpret_cast<PyArrayObject*>(data_obj.ptr());
	uint32_t* table = static_cast<uint32_t*>(PyArray_DATA(data_arr));
	uint32_t ncode = static_cast<uint32_t>(tncode);
	uint32_t sum = static_cast<uint32_t>(tsum);
	uint32_t symbol = static_cast<uint32_t>(tsymbol);
	//std::cout << coder->get_fname();
	coder->encode(table, ncode, sum, symbol);

}
uint32_t my_decoder(Coder * coder, boost::python::object data_obj, int tncode, int tsum) {
	PyArrayObject* data_arr = reinterpret_cast<PyArrayObject*>(data_obj.ptr());
	uint32_t* table = static_cast<uint32_t*>(PyArray_DATA(data_arr));
	uint32_t ncode = static_cast<uint32_t>(tncode);
	uint32_t sum = static_cast<uint32_t>(tsum);
	//std::cout << "decoding";
	uint32_t res=coder->decode(table, ncode, sum);
	//std::cout << res;
	return res;
}
void my_encoder2(Coder * coder, boost::python::object data_obj, int tncode, boost::python::object symbol_obj) {
	PyArrayObject* data_arr = reinterpret_cast<PyArrayObject*>(data_obj.ptr());
	PyArrayObject* label_arr = reinterpret_cast<PyArrayObject*>(symbol_obj.ptr());
	uint32_t* table = static_cast<uint32_t*>(PyArray_DATA(data_arr));
	uint32_t* label = static_cast<uint32_t*>(PyArray_DATA(label_arr));
	int i = 0;
	//std::cout << tncode;
	uint32_t ncode = static_cast<uint32_t>(tncode); 

	while (table[i*(tncode + 1)] < 1) {
		coder->encode(table+i*(tncode+1), ncode,table[i*(tncode+1)+tncode], label[i]);
		//std::cout << i << ":" << table[i*(tncode + 1) + tncode] << ":" << label[i] << std::endl;
		i += 1;
		
	}
}
void my_decoder2(Coder * coder, boost::python::object data_obj, int tncode, boost::python::object symbol_obj) {
	PyArrayObject* data_arr = reinterpret_cast<PyArrayObject*>(data_obj.ptr());
	PyArrayObject* label_arr = reinterpret_cast<PyArrayObject*>(symbol_obj.ptr());
	uint32_t* table = static_cast<uint32_t*>(PyArray_DATA(data_arr));
	float* label = static_cast<float*>(PyArray_DATA(label_arr));
	uint32_t ncode = static_cast<uint32_t>(tncode);
	//std::cout << "sizeof float" << sizeof(float) << std::endl;
	//std::cout << "decoding";
	int i = 0;
	while (table[i*(tncode + 1)] < 1) {
		label[i] = static_cast<float>(coder->decode(table + i*(tncode + 1), ncode, table[i*(tncode + 1) + tncode]));
		i += 1;
	}
	
}
BOOST_PYTHON_MODULE(Coder)
{

	using namespace boost::python;
	class_<Coder>("coder", init<std::string>())
		.def("encode", my_encoder)
		.def("decode", my_decoder)
		.def("encodes", my_encoder2)
		.def("decodes", my_decoder2)
		.def("start_encoder", &Coder::start_encoder)
		.def("end_encoder", &Coder::end_encoder)
		.def("start_decoder", &Coder::start_decoder)
		;
}