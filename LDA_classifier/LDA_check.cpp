#include <iostream>
#include "lda.hpp"

#include "data.hpp" // Macros in data.hpp file


int main(void)
{
BEGIN

  LDA<double> lda(DATAROWS, DATACOLS, 2);

  lda.copy_data(data);
  lda.fit(targets);
  lda.print_map();

END
  return 0;
}