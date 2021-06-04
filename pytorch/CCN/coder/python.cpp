#include <torch/extension.h>
#include"coder.h"

void my_encoder(Coder * coder, at::Tensor data_obj, int tncode, int tsum, int tsymbol) {
	uint32_t* table = static_cast<uint32_t *>(data_obj.to(torch::kCPU).to(torch::kInt32).data_ptr());
	uint32_t ncode = static_cast<uint32_t>(tncode);
	uint32_t sum = static_cast<uint32_t>(tsum);
	uint32_t symbol = static_cast<uint32_t>(tsymbol);
	//std::cout << coder->get_fname();
	coder->encode(table, ncode, sum, symbol);

}
uint32_t my_decoder(Coder * coder, at::Tensor data_obj, int tncode, int tsum) {
	uint32_t* table = static_cast<uint32_t *>(data_obj.to(torch::kCPU).to(torch::kInt32).data_ptr());
	uint32_t ncode = static_cast<uint32_t>(tncode);
	uint32_t sum = static_cast<uint32_t>(tsum);
	//std::cout << "decoding";
	uint32_t res=coder->decode(table, ncode, sum);
	//std::cout << res;
	return res;
}
void my_encoder2(Coder * coder, at::Tensor data_obj, int tncode,at::Tensor symbol_obj, int num) {
	int* table = data_obj.data_ptr<int>();
	int* label = symbol_obj.data_ptr<int>();
	int i = 0;
	//std::cout << tncode<<std::endl;
	uint32_t ncode = static_cast<uint32_t>(tncode); 
	uint32_t* ntable = new uint32_t[ncode+1];
	for (int i = 0; i<num; i++) {
		
		for(int j =0; j<tncode+1;j++){
			ntable[j] = static_cast<uint32_t>(table[i*(tncode + 1) + j]);
			//std::cout<<table[i*(tncode + 1) + j]<<" ";
		}
		coder->encode(ntable, ncode, ntable[ncode], static_cast<uint32_t>(label[i]));
		//std::cout<<"label:"<<label[i]<<std::endl;
		//std::cout << i << ":" << table[i*(tncode + 1) + tncode] << ":" << label[i] << std::endl;
	}
	delete[] ntable;
}
at::Tensor my_decoder2(Coder * coder, at::Tensor data_obj, int tncode, int num) {
	int* table = data_obj.data_ptr<int>();
	//uint32_t* table = static_cast<uint32_t *>(data_obj.to(torch::kCPU).to(torch::kInt32).data_ptr());
	auto option = data_obj.options().device(torch::kCPU).dtype(torch::kFloat32);
	at::Tensor symbol_obj = torch::empty({data_obj.size(0)},option);
	float* label = symbol_obj.data<float>();
	uint32_t ncode = static_cast<uint32_t>(tncode);
	uint32_t* ntable = new uint32_t[ncode+1];
	//std::cout << "sizeof float" << sizeof(float) << std::endl;
	//std::cout << "decoding";
	for (int i = 0; i<num;i++) {
		for(int j =0; j<tncode+1;j++){
			ntable[j] = static_cast<uint32_t>(table[i*(tncode + 1) + j]);
			//std::cout<<table[i*(tncode + 1) + j]<<" ";
		}
		label[i] = static_cast<float>(coder->decode(ntable, ncode, ntable[ncode]));
		//printf("decoding %dth symbol, it is %f\n", i, label[i]);
	}
	delete[] ntable;
	return symbol_obj;
}
namespace py = pybind11;
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	py::class_<Coder>(m,"coder")
        .def(py::init<std::string>())
        .def("encode", my_encoder)
		.def("decode", my_decoder)
		.def("encodes", my_encoder2)
		.def("decodes", my_decoder2)
		.def("start_encoder", &Coder::start_encoder)
		.def("end_encoder", &Coder::end_encoder)
		.def("start_decoder", &Coder::start_decoder);
};