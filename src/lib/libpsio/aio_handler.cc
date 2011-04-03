/*!
 \file
 \ingroup PSIO
 */

#include <cstdio>
#include <unistd.h>
#include <boost/shared_ptr.hpp>
#include <boost/thread/thread.hpp>
#include <libpsio/psio.h>
#include <libpsio/psio.hpp>

using namespace std;
using namespace boost;

namespace psi {

AIO_Handler::AIO_Handler(shared_ptr<PSIO> psio)
    : psio_(psio)
{
}
AIO_Handler::~AIO_Handler() {
}
shared_ptr<boost::thread> AIO_Handler::get_thread()
{
    return thread_;
}
void AIO_Handler::synchronize()
{
    thread_->join();
}
void AIO_Handler::read(unsigned int unit, const char *key, char *buffer, ULI size, psio_address start, psio_address *end) 
{
  unique_lock<mutex> lock(locked_);

  job_.push(1);
  unit_.push(unit);
  key_.push(key);
  buffer_.push(buffer);
  size_.push(size);
  start_.push(start);
  end_.push(end);

  if (job_.size() > 1) return;
  lock.unlock();

  //thread start
  thread_ = shared_ptr<boost::thread>(new boost::thread(boost::bind(&AIO_Handler::call_aio,this)));
}
void AIO_Handler::write(unsigned int unit, const char *key, char *buffer, ULI size, psio_address start, psio_address *end) 
{
  unique_lock<mutex> lock(locked_);

  job_.push(2);
  unit_.push(unit);
  key_.push(key);
  buffer_.push(buffer);
  size_.push(size);
  start_.push(start);
  end_.push(end);

  if (job_.size() > 1) return;
  lock.unlock();

  //thread start
  thread_ = shared_ptr<boost::thread>(new boost::thread(boost::bind(&AIO_Handler::call_aio,this)));
}
void AIO_Handler::read_entry(unsigned int unit, const char *key, char *buffer, ULI size) 
{
  unique_lock<mutex> lock(locked_);

  job_.push(3);
  unit_.push(unit);
  key_.push(key);
  buffer_.push(buffer);
  size_.push(size);

  if (job_.size() > 1) return;
  lock.unlock();

  //thread start
  thread_ = shared_ptr<boost::thread>(new boost::thread(boost::bind(&AIO_Handler::call_aio,this)));
}
void AIO_Handler::write_entry(unsigned int unit, const char *key, char *buffer, ULI size) 
{
  unique_lock<mutex> lock(locked_);

  job_.push(4);
  unit_.push(unit);
  key_.push(key);
  buffer_.push(buffer);
  size_.push(size);

  if (job_.size() > 1) return;
  lock.unlock();

  //thread start
  thread_ = shared_ptr<boost::thread>(new boost::thread(boost::bind(&AIO_Handler::call_aio,this)));
}
void AIO_Handler::read_discont(unsigned int unit, const char *key, 
  double **matrix, ULI row_length, ULI col_length, ULI col_skip, 
  psio_address start)
{
  unique_lock<mutex> lock(locked_);

  job_.push(5);
  unit_.push(unit);
  key_.push(key);
  matrix_.push(matrix);
  row_length_.push(row_length);
  col_length_.push(col_length);
  col_skip_.push(col_skip);
  start_.push(start);

  if (job_.size() > 1) return;
  lock.unlock();

  //thread start
  thread_ = shared_ptr<boost::thread>(new boost::thread(boost::bind(&AIO_Handler::call_aio,this)));
}
void AIO_Handler::write_discont(unsigned int unit, const char *key, 
  double **matrix, ULI row_length, ULI col_length, ULI col_skip, 
  psio_address start)
{
  unique_lock<mutex> lock(locked_);

  job_.push(6);
  unit_.push(unit);
  key_.push(key);
  matrix_.push(matrix);
  row_length_.push(row_length);
  col_length_.push(col_length);
  col_skip_.push(col_skip);
  start_.push(start);

  if (job_.size() > 1) return;
  lock.unlock();

  //thread start
  thread_ = shared_ptr<boost::thread>(new boost::thread(boost::bind(&AIO_Handler::call_aio,this)));
}
void AIO_Handler::call_aio()
{
  unique_lock<mutex> lock(locked_);

  while (job_.size() > 0) {
    lock.unlock();
    if (job_.front() == 1) { 
      psio_->read(unit_.front(),key_.front(),buffer_.front(),size_.front(),
        start_.front(),end_.front());

      lock.lock();
      job_.pop();
      unit_.pop();
      key_.pop();
      buffer_.pop();
      size_.pop();
      start_.pop();
      end_.pop();
      lock.unlock();
    }
    else if (job_.front() == 2) {
      psio_->write(unit_.front(),key_.front(),buffer_.front(),size_.front(),
        start_.front(),end_.front());

      lock.lock();
      job_.pop();
      unit_.pop();
      key_.pop();
      buffer_.pop();
      size_.pop();
      start_.pop();
      end_.pop();
      lock.unlock();
    }
    else if (job_.front() == 3) {
      psio_->read_entry(unit_.front(),key_.front(),buffer_.front(),
        size_.front());

      lock.lock();
      job_.pop();
      unit_.pop();
      key_.pop();
      buffer_.pop();
      size_.pop();
      lock.unlock();
    }
    else if (job_.front() == 4) {
      psio_->write_entry(unit_.front(),key_.front(),buffer_.front(),
        size_.front());

      lock.lock();
      job_.pop();
      unit_.pop();
      key_.pop();
      buffer_.pop();
      size_.pop();
      lock.unlock();
    }
    else if (job_.front() == 5) {

      double **A = matrix_.front();

      psio_address next_psio = start_.front();
      for (int i=0; i<row_length_.front(); i++) {
        psio_->read(unit_.front(),key_.front(),(char *) &(A[i][0]),
          sizeof(double)*col_length_.front(),next_psio,&next_psio);
        next_psio = psio_get_address(next_psio,sizeof(double)*
          col_skip_.front());
      }

      lock.lock();
      job_.pop();
      unit_.pop();
      key_.pop();
      matrix_.pop();
      row_length_.pop();
      col_length_.pop();
      col_skip_.pop();
      start_.pop();
      lock.unlock();
    }
    else if (job_.front() == 6) {

      double **A = matrix_.front();

      psio_address next_psio = start_.front();
      for (int i=0; i<row_length_.front(); i++) {
        psio_->write(unit_.front(),key_.front(),(char *) &(A[i][0]),
          sizeof(double)*col_length_.front(),next_psio,&next_psio);
        next_psio = psio_get_address(next_psio,sizeof(double)*
          col_skip_.front());
      }

      lock.lock();
      job_.pop();
      unit_.pop();
      key_.pop();
      matrix_.pop();
      row_length_.pop();
      col_length_.pop();
      col_skip_.pop();
      start_.pop();
      lock.unlock();
    }
    else {
      throw PsiException("Error in AIO: Unknown job type", __FILE__,__LINE__);
    }

    lock.lock();
  }
}

} //Namespace psi

